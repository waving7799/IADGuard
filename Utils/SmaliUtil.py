
FLAG = {'V': 'void',
        'Z': 'boolean',
        'B': 'byte',
        'S': 'short',
        'C': 'char',
        'I': 'int',
        'J': 'long',
        'F': 'float',
        'D': 'double'}

Default = {
        '[Z': 'boolean[]',
        '[B': 'byte[]',
        '[S': 'short[]',
        '[C': 'char[]',
        '[I': 'int[]',
        '[J': 'long[]',
        '[F': 'float[]',
        '[D': 'double[]'}

def deal_multi_default(samli_para_sig:str):
    par = []

    while not samli_para_sig.startswith("L"):
        if len(samli_para_sig) == 1:
            par.append(FLAG[samli_para_sig[0]])
            break
        elif len(samli_para_sig) == 0:
            break
        elif samli_para_sig.startswith("["):
            par_temp = parse_symbol(samli_para_sig)
            par.extend(par_temp)
            return par
            # par.append(Default[samli_para_sig[:2]])
            # samli_para_sig = samli_para_sig[2:]
        else:
            par.append(FLAG[samli_para_sig[0]])
            samli_para_sig = samli_para_sig[1:]

    par.append(samli_para_sig[1:].replace('/','.'))
    return par

def parse_symbol(part):
    par_temp = []
    for zd in Default:
        if part.startswith(zd):
            par_temp.append(Default[zd])
            part = part[2:]

    par = ""
    FLAG_list = False
    p = part.replace("/", ".")
    if p.startswith("["):
        FLAG_list = True
        p = p[1:]
    if p.startswith("L"):
        par = p[1:]
    elif p in FLAG:
        par = FLAG[p]
    if FLAG_list:
        par = par + "[]"
        par_temp.append(par)
    return par_temp

def make_params_class(part):
    cs = []
    obj = False
    array = 0
    start = 0
    tmp = part.split(";")
    for z1 in tmp:
        if z1.startswith('['):
            for zd in Default:
                if z1.startswith(zd):
                    cs.append(Default[zd])
                    z1 = z1[2:]

            par = ""
            FLAG_list = False
            p = z1.replace("/", ".")
            if p.startswith("["):
                FLAG_list = True
                p = p[1:]
            if p.startswith("L"):
                par = p[1:]
            elif p in FLAG:
                par = FLAG[p]
            if FLAG_list:
                par = par+"[]"

            cs.append(par)
        else:
            cs.extend(deal_multi_default(z1))

    return cs


def make_params_class_backup(part):
    cs = []
    obj = False
    array = 0
    start = 0
    tmp = part.split(";")
    for z1 in tmp:
        if z1.startswith('['):
            for zd in Default:
                if z1.startswith(zd):
                    cs.append(Default[zd])
                    z1 = z1[2:]
        for zd in FLAG:
            if z1.startswith(zd):
                cs.append(FLAG[zd])
                z1 = z1[1:]
                # break

        par = ""
        FLAG_list = False
        p = z1.replace("/", ".")
        if p.startswith("["):
            FLAG_list = True
            p = p[1:]
        if p.startswith("L"):
            par = p[1:]
        elif p in FLAG:
            par = FLAG[p]
        if FLAG_list:
            par = par+"[]"

        cs.append(par)

    return cs

def to_java_class_name(part, is_add=True):
    ret = '<error>'
    obj = False
    array = 0
    start = 0
    import_class = []
    for i in range(len(part)):
        if not obj:
            if part[i] == '[':
                array = array + 1
            elif part[i] == 'L':
                obj = True
                start = i
            else:
                ret = FLAG[part[i]]
                is_add = False
                break
        else:
            if part[i] == ';':
                cls = part[start + 1: i]
                cls = cls.replace('/', '.')
                ret = cls
                break
    if is_add:
        import_class.append(ret)

    ret = ret.split('.')[-1]

    while array > 0:
        ret = (ret + '[]')
        array = array - 1

    return ret


def smaliFucSig2Java(api, printflage=False):
    api = api[1:]
    tmp = api.split(";")
    class_name = tmp[0].replace("/", ".")

    rt = ""
    params = ""
    a = api.split(")")
    if len(a[1]) != 0:
        rt = a[1]
    a = a[0].split("(")
    tmp_params = api.split("(")[-1].split(")")[0]
    if len(a[1]) != 0:
        params = a[1]
        params = params.replace(" ", ";")

    func_name = tmp[1].split("(")[0].split(".")[-1]

    fun_types = make_params_class(tmp_params)
    fun_ret = make_params_class(rt)

    params = ",".join(fun_types)

    if params.endswith(","):
        params = params[:-1]
    java_method1 = "<" + class_name + ": " + fun_ret[0] + " " + func_name + \
                   "(" + params + ")>"

    if printflage:
        print(api, " ==> ", java_method1)

    return java_method1
