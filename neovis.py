#lol, this is where I'm putting visualization tools
import os 
import matplotlib.pyplot as plt
import re



def parse_idx_string(s):
    result = {}
    for ss in s.split(","):
        if len(ss) == 0: continue
        key,value = ss.split("=")
        result[key] = set([value])
    return result


import json
class Neovis():
    def __init__(self,path):
        self.incs = 'qwertyui'
        self.decs = 'asdfghjk'
        self.suffix = '_neovis.png'
        assert os.path.exists(path)
        self.path = path

        #zeropad al strings for sorting to make integer sorting work
        fs = sorted([x[:-len(self.suffix)] for x in os.listdir(path) if x.endswith(self.suffix)],key=lambda x : [xx.pop() for xx in parse_idx_string(x[:-len(self.suffix)]).values()])
        print(fs)
        #allkeys = [[x for x in parse_idx_string(f).values()] for f in fs]
        #[list(*parse_idx_string(f).values()) for f in fs]
        #print("fs",fs)
        self.fs = fs
        assert len(fs) > 0, "No files found"
        idxs = None
        for f in fs:
            idx = parse_idx_string(f)
            if idxs is None: idxs = idx
            else: assert set(idx.keys()) == set(idxs.keys()), "Keys in both dictionaries must match"
            idxs = {k: sorted(idx[k].union(idxs[k])) for k in idxs.keys()}
        assert len(idxs) > 0, "No indices found"
        assert len(idxs) <= len(self.incs), "Too many indices found, N=8 is current max"
        self.idxs = idxs
        #print(idxs)

    def make(self,num_frames=None,above_html="",anti_alias=True):
        imgs=''
        style_string = 'image-rendering: pixelated;' if not anti_alias else ""
        for f in self.fs:
            imgs += f'<div class="thing" id="{f}">{f}<br><img style="{style_string}" src="{f+self.suffix}"></div>\n'
        
        self.view_path = os.path.join(self.path,"view.html")

        vars = ''
        vars += 'const Dims = '+json.dumps(self.idxs) + ';\n'
        idx = {k:0 for k in self.idxs.keys()}
        vars += 'let Di = '+json.dumps(idx) + ';\n'
        vars += 'let inc2idx = '+json.dumps({self.incs[i]:k for i,k in enumerate(self.idxs.keys())}) + ';\n'
        vars += 'let dec2idx = '+json.dumps({self.decs[i]:k for i,k in enumerate(self.idxs.keys())}) + ';\n'
        html_path = os.path.join(os.path.dirname(__file__),"neovis.html")
        with open(html_path,'r') as f:
            self.html = f.read()

        inst = ''
        for i,k in enumerate(self.idxs.keys()):
            inst +=  f'<li>{self.incs[i]}/{self.decs[i]}: Increase/Decrease {k}. len({k}) = {len(self.idxs[k])}</li>\n'


        with open(self.view_path,"w") as f:
        #with open('neovis_out.html',"w") as f:
            txt=self.html.replace("%%above%%", above_html).replace("%%imgs%%",imgs)
            txt=txt.replace("%%vars%%",vars)
            txt=txt.replace("%%inst%%",inst)

            #print(txt)
            f.write(txt)


if __name__ == "__main__":
    n = Neovis("ignored/vis/im2")
    n.make()
    exit()


class Neovis_Bad():
    def __init__(self,indices=['a','b'],name='neovis',output_dir='py_grapher/'):
        self.name = name
        self.output_dir = output_dir
        os.makedirs(self.output_dir,exist_ok=True)
        self.indices_counts = {i:0 for i in indices}
        self.path = os.path.join(self.output_dir,self.name)
        self.framepath = os.path.join(self.path,'frames')
        os.makedirs(self.framepath,exist_ok=True)
        #self.framenames = os.path.join(self.framepath,self.name+'%d.png')
        #print(self.framepath)

    def save_plt_frame(self,frame_num = None,dpi = None, figure = None):
        num = self.frame_cnt if frame_num is None else frame_num
        if figure is None:
            figure = plt.gcf()
        figure.savefig(self.framenames%num,dpi=dpi)
        if frame_num is None:
            self.frame_cnt+=1
        else: 
            self.frame_cnt = frame_num
    
    def make(self,num_frames = None,above_html=""):
        imgs=''
        N = self.frame_cnt if num_frames is None else num_frames
        gf = lambda i : "frames/"+self.name+"%d.png"%i 
        for i in range(N):
            imgs += '<div class="thing">{num}<br><img src="{fn}"></div>\n'.format(fn=gf(i),num=i)
        self.view_path = os.path.join(self.path,"view.html")
        
        imgnum = max(list(map(lambda x : int(re.findall(f'{self.name}(\d+)',x)[0]),os.listdir(self.framepath))))
        self.last_frame_path = self.framenames%imgnum
        
        with open(self.view_path,"w") as f:
            txt=self.html.format(imgs=imgs,above=above_html)
            #print(txt)
            f.write(txt)
    
    def get_last_frame_path(self):
        try:
            imgnum = max(list(map(lambda x : int(re.findall(f'{self.name}(\d+)',x)[0]),os.listdir(self.framepath))))
        except Exception as e:
            print(self.framepath,self.name)
            raise e
        last_frame_path = f'{self.framepath}/{self.name}{imgnum}.png'
        return last_frame_path

    def redirect_at(self, path):
        rp = "./"+os.path.relpath(self.view_path,os.path.dirname(path))
        ht = f'<meta http-equiv="refresh" content="0; url={rp}" />' 
        with open(path,'w') as f:
            f.write(ht)

    html = """
    <html>
    <head>
    <style type="text/css">
    .thing {{display:none}}
    </style>
    </head>
    <body style="background-color:black;color:white">
    {above}

    {imgs}
    <script>
    var cur = 0;
    var n = document.querySelectorAll(".thing").length;
    function prev() {{
        cur -= 1;
        if (cur < 0) cur = 0;
        show();
    }}
    function next() {{
        cur += 1;
        if (cur >= n) cur = n-1;
        show();
    }}
    function first() {{
        cur=0;show();
    }}
    function last() {{
        cur=n-1;show();
    }}
    function show() {{
        document.querySelectorAll(".thing").forEach((a) => {{
            a.style.display = "none";
        }});
        document.querySelectorAll(".thing")[cur].style.display = "block";
    }}

    document.onkeydown = function(e) {{
        if (e.which == 37) prev();
        if (e.which == 39) next();
        if (e.which == 38) last();
        if (e.which == 40) first();
        e.preventDefault(); // prevent the default action (scroll / move caret)
    }};

    show();
    </script>

    </body>
    </html>
    """    