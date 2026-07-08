import asyncio
import time

from concurrent.futures import ThreadPoolExecutor, Future
from tqdm.auto import tqdm
from dataclasses import dataclass, field

import nlplib.utils.io as io


async def process_prompt(messages: str | list[dict], client, semaphore) -> str:
    if isinstance(messages, str):
        messages=[{"role": "user", "content": messages}]
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="minimax-m2.7",
                messages=messages,
                n=1,
                top_p=0.95,
                temperature=0.9,
                max_tokens=4000,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }  
                }
            )
            return response
        except Exception as e:
            return f"Ошибка: {str(e)}"


def on_done(data_idx, results, out_file=None):
    print(f'on_done data_idx={data_idx}')
    if results is None:
        return
    compls = [
        (i, res.choices[0].message.content)
        for i, res in enumerate(results)
        if not (isinstance(res, str) or res is None)]
    print(f'Данные #{data_idx} выполнено задач {len(compls)}')
    io.write_json(compls, out_file)


@dataclass
class InferData:
    """
    вариант с отдельным потоком (чтобы не блокировать Jupyter)
    """
    idx: int
    data: list
    results: list | None = field(init=False, default=None)
    done: bool = False
    on_done: object = None     # callback on future done

    @property
    def remaining(self):
        """
        Ещё не обработанные сэмплы с их оригинальными индексами.
        """
        if self.results is None:
            self.results = [None]*len(self.data)
            return list(enumerate(self.data))
        remaining_ = [
            (i, sample) for i, (sample, res) 
            in enumerate(zip(self.data, self.results)) if res is None or isinstance(res, str)]
        if len(remaining_) == 0:
            self.done = True
            return None
        return remaining_


class ThreadedInferer:
    def __init__(self, max_concurrent=5):
        self.max_concurrent = max_concurrent
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures: dict[int, Future] = {}
        self.data_list: dict[int, InferData] = {}
        self._counter = 0
        self._stop = False

    def stop(self):
        # текущий батч доработает, следующие — нет
        self._stop = True
        print("Сигнал остановки отправлен.")

    def get_results(self, data_idx: int | None = None):
        if data_idx is None:
            for i, future in self._futures.items():
                if future and not future.done():
                    print(f"Данные #{i} ещё обрабатываются...")
            return {idx: data.results for idx, data in self.data_list.items()}
        data = self.data_list.get(data_idx)
        if data is None:
            return None
        future = self._futures.get(data_idx)
        if future and not future.done():
            print(f"Данные #{data_idx} ещё обрабатываются...")
        return data.results

    def wait(self, data_idx: int | None = None):
        if data_idx is not None:
            self._futures[data_idx].result()    # ждем завершения
            return self.data_list[data_idx].results

        self._executor.shutdown(wait=True, cancel_futures=False)
        self._executor = ThreadPoolExecutor(max_workers=1)          # пересоздаем на будущее

        return {idx: data.results for idx, data in self.data_list.items()}

    def _register_callback(self, idx, future):
        on_done = self.data_list[idx].on_done
        if on_done is not None:
            future.add_done_callback(
                lambda f, i=idx: on_done(i, self.data_list[i].results)
            )

    def enqueue(self, data, max_seconds=600, on_done_callback=None) -> int:
        self._stop = False
        idx = self._counter
        self._counter += 1
        inf_data = InferData(idx=idx, data=data, on_done=on_done_callback)
        future = self._executor.submit(self._run, idx, inf_data, max_seconds)
        self._futures[idx] = future
        self.data_list[idx] = inf_data
        self._register_callback(idx, future)
            
        print(f"Данные #{idx} добавлены ({len(data)} задач)")
        return idx

    def resume(self, max_seconds=600) -> int:
        self._stop = False
        remaining_data = [(i, data) for i, data in self.data_list.items() if not data.done]
        if not remaining_data:
            print('Нет незавершенных данных')
            return
        for i, d in remaining_data:
            self._futures[i] = self._executor.submit(self._run, i, d, max_seconds)
            self._register_callback(i, self._futures[i])
            print(f"Данные #{i} продолжены")     


    def _run(self, idx, data, max_seconds):
        if self._stop:
            print(f"Данные #{idx} пропущены (остановка).")
            return None
        return asyncio.run(self._async_runner(data, max_seconds))
        
    async def _async_runner(self, data, max_seconds):
        START_TIME = time.time()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        remaining = data.remaining
        if remaining is None:
            return data.results
        print(f"Осталось {len(remaining)} задач")
        ids, remaining = list(zip(*remaining))
         
        tasks = [asyncio.create_task(process_prompt(sample, semaphore)) for sample in remaining]
        progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Генерация")
        try:
            for coro in progress:
                if self._stop or time.time() - START_TIME > max_seconds:
                    break
                try:
                    await coro
                except Exception:
                    pass
        finally:
            progress.close()
            cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
            await asyncio.gather(*tasks, return_exceptions=True)
        for idx, task in zip(ids, tasks):
            if task.done() and not task.cancelled() and task.exception() is None:
                data.results[idx] = task.result()
            elif task.cancelled():
                data.results[idx] = "Отменено"
            elif task.done():
                data.results[idx] = f"Ошибка: {task.exception()}"
            else:
                data.results[idx] = "Не успело стартовать"
        return data.results