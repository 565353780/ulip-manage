def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])

    if total >= 1e9:
        return "{:.2f}B".format(total / 1e9)
    elif total >= 1e6:
        return "{:.2f}M".format(total / 1e6)
    elif total >= 1e3:
        return "{:.2f}K".format(total / 1e3)
    else:
        return str(total)
