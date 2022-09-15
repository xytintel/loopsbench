import json
import argparse
import os
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser(description='painter')

parser.add_argument('--xs', type=str, required=True) # sort dim
parser.add_argument('--ys', type=str, required=True)
parser.add_argument('--title', type=str, required=True)
parser.add_argument('--files', type=str, required=True)
parser.add_argument('--sortx', action='store_true', default=False)
parser.add_argument('--begin', type=str, default='-------------------- output --------------------')
args = parser.parse_args()

def read_jsonlines(filename):
    lines = []
    with open(filename, 'r') as f:
        flag = False
        for line in f.readlines():
            if args.begin in line:
                flag = True
            if flag:
                lines.append(line.strip())
    lines = lines[1:]
    lines = [json.loads(line.replace('\'', '\"')) for line in lines]
    return lines

def get_key(input):
    xs = input[1]
    return float(xs)

lineinfos = []
for file in glob.glob(args.files):
    basename = os.path.basename(file)
    basename = basename[:basename.rfind('.')]
    jls = read_jsonlines(file)
    lines = [[basename, jl[args.xs], jl[args.ys]] for jl in jls]
    if args.sortx:
        lines = sorted(lines, key=get_key)
    lineinfos.append(lines)

plt.grid(True)
plt.xlabel(args.xs)
plt.ylabel(args.ys)
plt.title(args.title)

for idx, lineinfo in enumerate(lineinfos):
    xs = []
    ys = []
    for point in lineinfo:
        xs.append(point[1])
        ys.append(point[2])
    plt.plot(xs, ys)
plt.legend([k[0][0] for k in lineinfos])
plt.gcf().autofmt_xdate()
# plt.show()
plt.savefig(os.path.join(args.title + '.jpg'))
plt.close()
