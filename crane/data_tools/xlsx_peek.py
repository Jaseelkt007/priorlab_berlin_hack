import zipfile, re, sys
def rows(path, n=6):
    z = zipfile.ZipFile(path)
    ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        ss = [re.sub('<[^>]+>', '', m) for m in re.findall(r'<si>(.*?)</si>', z.read('xl/sharedStrings.xml').decode())]
    sheets = sorted(x for x in z.namelist() if x.startswith('xl/worksheets/sheet'))
    out = []
    for sh in sheets:
        x = z.read(sh).decode()
        R = re.findall(r'<row[^>]*>(.*?)</row>', x)
        def cells(r):
            vals = []
            for attrs, v in re.findall(r'<c([^>]*)>(?:<f>.*?</f>)?<v>(.*?)</v></c>', r):
                vals.append(ss[int(v)] if 't="s"' in attrs else v)
            return vals
        out.append((sh, len(R), [cells(r) for r in R[:n]]))
    return out
for p in sys.argv[1:]:
    print('==', p.split('/')[-1])
    for sh, nr, rs in rows(p):
        print(' ', sh, 'rows', nr)
        for r in rs: print('   ', r[:12])
