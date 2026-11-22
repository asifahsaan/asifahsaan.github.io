"""Assemble a lab page: python build.py cnn   ->  ../visual-learning/cnn.html"""
import sys, pathlib
here = pathlib.Path(__file__).parent
def rd(n): return (here / n).read_text(encoding='utf-8')
def build(name):
    t = rd(f'{name}.template.html')
    t = t.replace('/*KATEX_CSS*/', rd('katex.css.part') + '\n</style>')
    t = t.replace('/*COMMON_CSS*/', rd('common.css'))
    t = t.replace('/*KATEX_JS*/', rd('katex.js.part'))
    t = t.replace('/*COMMON_JS*/', rd('common.js'))
    t = t.replace('/*PANEL_JS*/', rd('panel.js'))
    import re
    t = re.sub(r'/\*INCLUDE:([\w.]+)\*/', lambda mo: rd(mo.group(1)), t)
    if '/*DEFAULTS_JS*/' in t: t = t.replace('/*DEFAULTS_JS*/', rd(f'{name}_defaults.js'))
    out = here.parent / 'visual-learning' / ({'nn': 'neural-network'}.get(name, name) + '.html')  # rnn -> rnn.html
    out.write_text(t, encoding='utf-8'); print(out, len(t))
for n in sys.argv[1:]: build(n)
