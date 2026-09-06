import torch, gc, sys, types, weakref, traceback
from IPython import get_ipython


def collect():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()  # ipc_collect() нужен только при разделении тензоров между процессами
        torch.cuda.empty_cache()  # empty_cache() возвращает драйверу только блоки, на которые уже нет ссылок, поэтому gc.collect() должен идти первым.
        torch.cuda.synchronize()


def count_models():
    from collections import Counter
    c = Counter()
    for x in gc.get_objects():
        try:
            nm = type(x).__name__
            if nm in ('SentenceTransformer', 'CrossEncoder', 'Transformer', 'XLMRobertaModel') \
               and isinstance(x, torch.nn.Module):
                c[nm] += 1
        except Exception:
            continue
    return c


def allocated_tensors(top=15, show_holders=0):
    """
    аллоцированная память под тензоры (поиск через ссылки gc) и их держатели.
    По формам сразу видно природу: тензоры, совпадающие по размеру с весами модели 
    — это градиенты или состояния Adam; 
    что-то вида [batch, seq, 1024] — активации.

    Notes:
    - части тензоров get_referrers вернёт пусто или что-то бесполезное — например, param.grad хранится 
      на стороне C++, и питоновский GC его связь не всегда показывает. Тогда ориентируйтесь по форме: 
      совпадает с формой какого-то веса → это градиент или момент оптимизатора.
    - Пример:
      Состояния Adam опознаются так: exp_avg лежит в словаре {'step':…, 'exp_avg':…, 'exp_avg_sq':…}, 
      владелец которого — defaultdict внутри optimizer.state.
      если у тензора владелец — dict с ключами вроде ['weight'], а над ним Module, это штатный параметр модели, всё в порядке.
      Тревожный признак — frame (traceback жив) или defaultdict с exp_avg (оптимизатор не освободился).
    - list среди владельцев сам по себе ни о чём не говорит: длинный list на сотни тысяч элементов —
      это чей-то снимок gc.get_objects(), он ссылается на всю кучу. Свой собственный снимок функция
      отфильтровывает, чужие (из вашей ячейки) — нет, проверяйте len().
    """
    # снимок держим в переменной, а не в временном выражении: иначе он попадает в держатели
    # каждого тензора как безымянный list на сотни тысяч элементов и сбивает диагностику
    all_objs = gc.get_objects()
    storages = {}
    for o in all_objs:
        try:
            if not (torch.is_tensor(o) and o.is_cuda):
                continue
            st = o.untyped_storage()
        except Exception:
            continue
        key = st.data_ptr()
        if key not in storages:
            storages[key] = [st.nbytes(), []]
        storages[key][1].append(o)

    items = sorted(storages.values(), key=lambda x: -x[0])
    total = sum(n for n, _ in items)

    for n, ts in items[:top]:
        t = ts[0]
        print(f'{n/2**20:8.1f} MiB  {tuple(t.shape)}  {t.dtype}  '
              f'req_grad={t.requires_grad}  views={len(ts)}')

    print(f'\nуникальных хранилищ: {len(items)}, суммарно {total/2**30:.2f} GiB')
    print(f'memory_allocated: {torch.cuda.memory_allocated()/2**30:.2f} GiB')

    # ---- держатели: только для топ-N, по запросу ----
    # Фильтр mine. Кроме контейнеров здесь добавлен фрейм самой функции — иначе первым же держателем 
    # каждого тензора будет frame, потому что t и ts это локальные переменные. Список ts добавляется в mine лениво, 
    # прямо в цикле, чтобы не обходить все items заранее.
    myframe = sys._getframe()
    mine = {id(storages), id(items), id(myframe), id(all_objs)}
    skip = lambda r: id(r) in mine or r is myframe

    for n, ts in items[:show_holders]:
        mine.add(id(ts))
        t = ts[0]
        print(f'\n--- держатели {tuple(t.shape)}, {n/2**20:.1f} MiB ---')
        # refs держим в переменной: пока идёт этот цикл, список жив и содержит r, поэтому
        # вложенный get_referrers(r) вернёт его же — тот самый фантомный «владелец: list»
        refs = gc.get_referrers(t)
        for r in refs:
            if skip(r):
                continue
            if isinstance(r, dict):
                ks = list(r.keys())[:5]
                kinds = {type(k).__name__ for k in ks}
                if 'weakref' in kinds:
                    print(f'  dict, {len(r)} ключей, типы: {kinds}, пример: {ks[:2]}')
                else:
                    print('  dict, ключи:', ks)
                for owner in gc.get_referrers(r):
                    if not skip(owner) and owner is not refs:
                        print('    владелец:', type(owner).__name__)
            else:
                if type(r).__name__ == 'frame':
                    print(f'  frame {r.f_code.co_name} '
                          f'({r.f_code.co_filename.split("/")[-1]}:{r.f_lineno})')
                else:
                    print(' ', type(r).__name__, str(r)[:120].replace('\n', ' '))

    all_objs = None      # снимок ссылается на всю кучу — без этого gc.collect() бесполезен
    gc.collect()


def who_holds(obj, max_depth=10, max_nodes=300_000):
    """Печатает 'имя_переменной -> контейнеры -> obj' — кто реально держит объект."""
    ip = get_ipython()
    ns = ip.user_ns if ip is not None else {}
    oh = ns.get('_oh')
    me = sys._getframe()
    queue = [(obj, [])]
    mine = {id(queue)}
    seen = {id(obj)}
    found, nodes = [], 0

    while queue and nodes < max_nodes:
        cur, path = queue.pop(0)
        if len(path) >= max_depth:
            continue
        for r in gc.get_referrers(cur):
            nodes += 1
            if r is me or isinstance(r, types.FrameType) or id(r) in mine:
                continue
            if isinstance(r, tuple) and len(r) == 2 and r[0] is cur:
                continue                                  # элемент нашей очереди
            if r is ns:
                for k, v in ns.items():
                    if v is cur:
                        found.append((k, path))
                continue
            if oh is not None and r is oh:
                for k, v in oh.items():
                    if v is cur:
                        found.append((f'Out[{k}]', path))
                continue
            if id(r) in seen:
                continue
            seen.add(id(r))
            queue.append((r, [type(r).__name__] + path))

    if not found:
        print('корень не найден: держит C++/фрейм, либо мал max_depth')
    for name, path in found:
        print(f'  {name} -> ' + ' -> '.join(path or ['<сам объект>']))
    return found


def who_holds_biggest():
    best = None
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                n = o.untyped_storage().nbytes()
                if best is None or n > best[0]:
                    best = (n, o)
        except Exception:
            continue
    if best:
        print(f'{best[0]/2**20:.1f} MiB {tuple(best[1].shape)}')
        return who_holds(best[1])


def _resolve(target, ns):
    """('имя' | сам объект) -> (имя_для_удаления | None, объект | None)"""
    if isinstance(target, str):
        if target not in ns:
            print(f'  {target}: имени нет в user_ns — пропускаю')
            return target, None
        return target, ns[target]
    return None, target


def deep_free(*targets, objs=()):
    """
    освободить аллоцированную память под тензоры

    targets — ИМЕНА переменных ('model', 'trainer', 'loss') либо сами объекты.
    Имена предпочтительнее: deep_free('model') сам удалит имя из user_ns и не создаст
    лишней сильной ссылки в своём фрейме. Вариант deep_free(model) такую ссылку создаёт,
    а после `del model` вообще падает с NameError, то есть не отрабатывает вовсе.

    Объекты после deep_free непригодны к использованию: у них обнуляются .model, .optimizer,
    .cache и прочие ссылки. Это не «мягкая очистка», а разрушение.

    Кто именно держит память:
    1. sys.last_traceback: при исключении или KeyboardInterrupt IPython сохраняет traceback,
       а тот держит все фреймы стека со всеми локальными переменными: trainer, optimizer,
       текущий батч, активации.
    2. Кэш выводов IPython. Out[53], _, __, _53 держат результаты ячеек. Плюс _oh, _ih.
       Сюда попадает всё, что было результатом ячейки, — например `model[0].auto_model`
       без print() держит весь backbone. Отдельно ip.user_ns_hidden: displayhook кладёт
       результат и туда, а flush() его не чистит — самый незаметный держатель, объект
       не виден ни в %who, ни в Out. И ip.last_execution_result — результат последней ячейки.
    3. Градиенты параметров модели (.grad).
    4. AcceleratorState — Trainer создаёт её как глобальный синглтон, она может держать
       ссылки на модель и оптимизатор.
    5. Кэш лоссов (например у CachedMultipleNegativesRankingLoss).
    6. Сильная ссылка в обычном глобальном контейнере: dict с гиперпараметрами, список
       моделей и т.п. Их deep_free не видит: он лишь сообщит, что объект пережил удаление,
       и вернёт weakref'ы выживших. Держателя ищите вручную через who_holds — автоматически
       он не вызывается, потому что стоит десятки секунд: каждый gc.get_referrers обходит
       всю кучу, порядка 35 с на 20k узлов, а дефолтные max_nodes=300_000 — это минуты.

           alive = deep_free('model', 'trainer')
           who_holds(alive[0][1](), max_nodes=20_000)

    Ориентир по allocated_tensors: если остались только веса (нет дублей тех же форм и нет
    dict с exp_avg/exp_avg_sq), то оптимизатор и градиенты уже освободились и держат саму
    модель — ищите пункт 6.
    """
    ip = get_ipython()
    ns = ip.user_ns if ip is not None else {}

    resolved = [_resolve(t, ns) for t in (*targets, *objs)]

    # 1. traceback прерывания — все варианты имён.
    # Обнулить sys.last_* мало: если на тот же traceback ссылается что-то ещё (сохранённое
    # исключение, цепочка __context__/__cause__), фреймы выживут вместе со своими локальными
    # списками — например device_exp_avgs внутри foreach-версии AdamW. clear_frames() чистит
    # f_locals у всех фреймов traceback'а и рвёт это независимо от того, кто его держит.
    for n in ('last_traceback', 'last_value', 'last_type', 'last_exc'):
        obj = getattr(sys, n, None)
        if obj is not None:
            tb = obj if isinstance(obj, types.TracebackType) else getattr(obj, '__traceback__', None)
            if tb is not None:
                try:
                    traceback.clear_frames(tb)
                except Exception:
                    pass
        if hasattr(sys, n):
            try:
                setattr(sys, n, None)
            except Exception:
                pass
    obj = tb = None

    # Каждый шаг по IPython — под своим try. displayhook.flush() кидает ValueError при
    # do_full_cache=False, и без try обрывал бы всё остальное в функции, включая градиенты
    # и AcceleratorState.
    if ip is not None:
        try:
            itb = getattr(ip, 'InteractiveTB', None)
            if getattr(itb, 'tb', None) is not None:
                traceback.clear_frames(itb.tb)
            if itb is not None:
                itb.tb = None              # ultratb держит свою копию
        except Exception as e:
            print(f'  InteractiveTB: {type(e).__name__}: {e}')
        try:
            ip.displayhook.flush()         # Out[...], _1.._N
        except Exception as e:
            print(f'  displayhook.flush(): {type(e).__name__}: {e}')
            oh = ns.get('_oh')             # руками, раз штатный путь не сработал
            if isinstance(oh, dict):
                oh.clear()
            for k in [k for k in ns if k[:1] == '_' and k[1:].isdigit()]:
                ns.pop(k, None)            # _12 и прочие результаты ячеек
        for k in ('_', '__', '___', '_i', '_ii', '_iii'):
            ns.pop(k, None)
        # user_ns_hidden — вторая копия тех же имён: displayhook пишет результат ячейки
        # и в user_ns, и сюда (shell.push(..., interactive=False)), а flush() чистит только
        # user_ns. Поэтому _10, _84 и прочие результаты остаются здесь навсегда — если
        # ячейка вернула модель (model.eval(), model.to(...) возвращают сам модуль),
        # она будет жить до конца сессии, не показываясь ни в %who, ни в Out.
        hidden = getattr(ip, 'user_ns_hidden', None)
        if isinstance(hidden, dict):
            for k in [k for k in hidden
                      if k.strip('_') == '' or (k[:1] == '_' and k[1:].isdigit())]:
                hidden.pop(k, None)
        # результат последней ячейки лежит отдельно от Out, flush() его не трогает.
        # Опасно тем, что там может оказаться список от gc.get_objects(): такой снимок
        # ссылается на каждый объект процесса и в одиночку держит вообще всё.
        res = getattr(ip, 'last_execution_result', None)
        if res is not None:
            try:
                res.result = None
            except Exception:
                pass

    # 2. кэши лосса / трейнера и градиенты
    o = None
    for _, o in resolved:
        if o is None:
            continue
        for a in ('cache', 'random_states', 'embeddings', 'optimizer', 'lr_scheduler', 'model'):
            if hasattr(o, a):
                try:
                    setattr(o, a, None)    # trainer.model / loss.model — сильные ссылки на модель
                except Exception:
                    pass
        # optimizer.state — defaultdict {param: {'step','exp_avg','exp_avg_sq'}}, у AdamW это
        # две копии весов. Чистим по месту: так память уходит даже если сам оптимизатор
        # держит кто-то ещё. param_groups в условии — чтобы не задеть чужой атрибут .state.
        state = getattr(o, 'state', None)
        if isinstance(state, dict) and hasattr(o, 'param_groups'):
            try:
                state.clear()
            except Exception:
                pass
        state = None
        zero_grad = getattr(o, 'zero_grad', None)
        if callable(zero_grad):
            try:
                zero_grad(set_to_none=True)
            except Exception:
                pass
    o = zero_grad = None

    # 3. удаляем имена, а сами объекты держим только через weakref — иначе сильные ссылки
    # остались бы в этом фрейме и gc.collect() ниже ничего бы не собрал
    alive = []
    for name, o in resolved:
        ref = None
        if o is not None:
            try:
                ref = weakref.ref(o)
            except TypeError:              # dict/list/tuple weakref не поддерживают
                pass
        if name is not None:
            ns.pop(name, None)
        if ref is not None:
            alive.append((name, ref))
    resolved = o = ref = None

    try:
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:
        pass

    gc.collect()
    collect()

    # 4. кто из целей пережил удаление. who_holds отсюда не зовём — он стоит десятки секунд;
    # возвращаем weakref'ы, чтобы натравить его вручную на конкретный объект.
    survived = [(name, ref) for name, ref in alive if ref() is not None]
    for name, ref in survived:
        print(f'!!! {name or type(ref()).__name__} пережил удаление — держит кто-то ещё')
    if survived:
        print('    держателя искать так: who_holds(alive[0][1](), max_nodes=20_000)')

    print(f'{torch.cuda.memory_allocated()/2**30:.2f} / '
          f'{torch.cuda.memory_reserved()/2**30:.2f} GiB')
    return survived
