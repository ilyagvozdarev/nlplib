import torch


def check_mixed_precision_break(model, texts):
    # тестирование mixed precision на самом длинном текстовом входе
    max_text = max(texts.values(), key=lambda p: len(p))

    bad = []
    def mk(name):
        def f(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.is_floating_point():
                bad.append((name, 'NON-FINITE' if not torch.isfinite(t).all()
                                else t.abs().max().item()))
        return f

    hs = [m.register_forward_hook(mk(n)) for n, m in model.named_modules()]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        model.encode([max_text], convert_to_tensor=True)
    for h in hs: 
        h.remove()

    first = next((n for n, v in bad if v == 'NON-FINITE'), None)
    print('ПЕРВЫЙ СЛОМ:', first)
    print('максимумы:', sorted([(v, n) for n, v in bad if v != 'NON-FINITE'], reverse=True)[:5])