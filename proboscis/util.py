from multiprocessing.pool import Pool

n__author__ = 'kentomasui'
import multiprocessing
import os
import sys
import zipfile
from functools import reduce
from itertools import islice
import hashlib
import dill

sys.setrecursionlimit(10000000)

class infix(object):
    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        self.left = other
        return self

    def __or__(self, other):
        return self.function(self.left, other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


def lmap(f, *iterable): return list(map(f, *iterable))


@infix
def compose(f, g):
    """
    :param f:
    :type f: (t)->u
    :param g:
    :type g: (s)->t
    :return:
    :rtype:
    """
    return lambda t: f(g(t))


@infix
def and_then(f, g):
    return compose(g, f)


def par_map(f, iterable):
    pool = multiprocessing.Pool()
    return pool.map(f, iterable)


def mk_string(seq, separator):
    return reduce(lambda a, b: str(a) + separator + str(b), seq, "")


class dotdict(dict):
    """
    let dictionary values be accessed(get) via dot accessor like:
    value = dict.key
    """

    def __getattr__(self, attr):
        res = self.get(attr)
        if (type(res) == dict):
            return dotdict(res)
        else:
            return res

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dict_to_function(dict):
    """
    :param dict:
    :type dict: dict[K,V]
    :return: function backed by dict
    :rtype: (K)->V
    """

    def l(k):
        return dict[k]

    return l


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def autoClose(filename, param, f):
    with open(filename, param) as F:
        return f(F)


def readAppend(filename, f):
    path = os.path.expanduser(filename)
    files = [open(path, p) for p in ['r', 'a']]
    f(files[0], files[1])
    for f in files:
        f.close()


def fileLines(file_name):
    """
    open fileName, and generates lines of file.
    :param file_name:
    :type file_name: str
    :return: generator of file lines
    """
    with open(file_name, 'r') as f:
        for line in f:
            yield line


def fileString(fileName):
    """
    :param fileName:
    :return: whole file as a string object
    """
    return reduce(lambda a, b: a + b, list(fileLines(fileName)))


def zipFileLines(zipName, fileName):
    """
    :param zipName: zipped file
    :param fileName: file in zipped file
    :return: lines of file
    """
    with zipfile.ZipFile(zipName) as z:
        with z.open(fileName) as f:
            for line in f:
                yield line


def ensurePathExists(fileName):
    """
    make directory if not present
    :param fileName:
    :return:None
    """
    from os import path, makedirs
    parent = os.path.dirname(fileName)
    if not path.exists(parent) and parent:
        makedirs(parent)


def exists(filename):
    from os import path
    return path.exists(filename)


def fileMemo(f, path):
    """
    convert a function to cache its return value at path
    :param f: src function
    :param path: path to save cache
    :return: new function that caches its return value at path
    """
    if exists(path):
        print("loaded cache: " + path)
        cache = load(path)
    else:
        print("cannot find cache: " + path)
        cache = {}

    def l(param):
        key = frozenset(list(param.items()))
        print(key)
        for pair in key:
            print(pair)
        if key in cache:
            return cache[key]
        else:
            cache[key] = f(param)
            save(cache, path)
            return cache[key]

    return l


def saveIfNotExist(path, f, force=False):
    if exists(path) and not force:
        print("path exists, and performing again: " + path)
        return f()
    else:
        data = f()
        print("saved data: " + path)
        save(data, path)
        return data


def loadOrCall(path, proc, force=False):
    """
    load cached data from path if present.
    otherwise, proc will be called and saved to path,
    so that it can be loaded when this function is called again.
    :type path: str
    :type proc: ()->T
    :type force:Bool
    :param path: path to save/load data
    :param proc: a function that returns value you want to cache
    :param force: if true, invalidate the cache and call proc
    :return: return value of proc or loaded cache
    :rtype: T
    """
    if exists(path) and not force:
        print("loading cache: " + path)
        time, res = check_time(lambda: load(path))
        print("loading cache took {0} seconds.".format(time))
        return res
    else:
        if force:
            print("forcing to calculate data! =>", path)
        else:
            print("cache is not found! recalculating", path)
        time, data = check_time(proc)
        print("procedure took {0} seconds.".format(time))
        print("saved cache: " + path)
        save(data, path)
        return data

def save(obj, filename:str):
    """
    save object at fileName using pickle serializer
    :param obj:
    :param filename:
    :return: filename
    """
    ensurePathExists(filename)
    autoClose(filename, 'wb', lambda f: dill.dump(obj, f))

def save_as_hash(obj,file_dir:None):
    """
    save object as sha1+.pkl using dill.
    you probablly need dill to load this object.
    or just use util.load()
    :param obj:
    :param file_dir:
    :return: new_name
    """
    import tempfile,shutil
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        dill.dump(obj,tmp)
        filename = os.path.join(file_dir,tmp.name)
    new_name = sha1(filename)+".pkl"
    # hope this works with full path
    shutil.copy(filename,os.path.join(file_dir,new_name))
    return new_name

def sha1(filename):
    with open(filename, "rb") as f:
        data = f.read()
        return hashlib.sha1(data).hexdigest()

def load(fileName, print_time=True):
    """
    load object from fileName using pickle
    :param fileName:
    :return: obj
    """
    if print_time:
        def l():
            print("loading file:" + fileName)
            res = autoClose(fileName, 'rb', lambda f: dill.load(f))
            print("done.")
            return res

        t, res = check_time(l)
        print("loading file took {0:.3f} seconds".format(t))
        return res
    else:
        return autoClose(fileName, 'rb', lambda f: dill.load(f))


def check_time(f):
    """
    :param f:
    :type f: ()->T
    :return:
    :rtype: long,T
    """
    import time
    start = time.time()
    result = f()
    end = time.time()
    return (end - start), result


if __name__ == '__main__':
    from itertools import groupby

    data = [1.1, 2.1, 3.1, 4.1, 5.1, 1.2]
    groups = groupby(data, lambda a: frozenset([a, a]))
    for k, v in groups:
        print(k, v)


def makeImage(data, resolution, tileShape):
    from utils import tile_raster_images
    try:
        import PIL.Image as Image
    except ImportError:
        import Image
    return Image.fromarray(
        tile_raster_images(X=data,
                           img_shape=resolution, tile_shape=tileShape,
                           tile_spacing=(1, 1)))


def run_command(command):
    import subprocess
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')


def writeFile(path, f):
    autoClose(path, "w", f)


def writeFileStr(path, string):
    autoClose(path, "w", lambda f: f.write(string))

def write_file_str(path,string:str):
    ensurePathExists(path)
    autoClose(path,"w",lambda f:f.write(string))

def partition(predicate, seq):
    """
    :param predicate:
    :param seq: traversable object
    :return: tuple of list that satisfies given predicate, and else
    predicate must return ether True or False given element of seq.
    """
    a = []
    b = []
    for e in seq:
        if predicate(e):
            a.append(e)
        else:
            b.append(e)
    return a, b


def group_by(iterable, key=lambda e: e):
    """
    :param iterable:
    :param key:
    :return: groups made of list
    """
    from itertools import groupby
    l = sorted(list(iterable), key=key)
    return {k: list(g) for k, g in groupby(l, key)}


def take_while(iterable, pred):
    for e in iterable:
        if pred(e):
            yield e
        else:
            break


def iterate(start, succ):
    """
    :type start: T
    :type succ: (T) -> T
    :param start:
    :param succ:
    :return:
    """
    var = start
    while True:
        yield var
        var = succ(var)


def any_range(start, end, step):
    """
    :type start: T
    :type end: T
    :param start:
    :param end:
    :param step: anything that can be added to T
     :rtype:generator of T
    """
    var = start
    while var < end:
        yield var
        var = var + step


def any_sum(iterable, initial=None):
    """
    :type iterable:iterable of T
    :param iterable: anything that have __add__ defined for T
    :return: sum of iterable
    :rtype: T
    """
    if initial is None:
        return reduce(lambda a, b: a + b, iterable)
    else:
        return reduce(lambda a, b: a + b, iterable, initial)


def map_values(f, dict):
    """
    :param f:
    :type f: (R)->S
    :param dict:
    :type dict: dict[P,R]
    :return:
    :rtype:dict[P,S]
    """
    return {k: f(v) for k, v in list(dict.items())}


class MappedDict:
    def __init__(self, f, dict):
        self.f = f
        self.dict = dict

    def __getitem__(self, item):
        return self.f(self.dict[item])


def map_values_lazy(f, dict):
    return MappedDict(f, dict)


def shuffle(iterable):
    """
    """
    import random
    shuffled = list(iterable)
    random.shuffle(shuffled)
    return shuffled


def find_first(iterable, predicate):
    """
    :param iterable:
    :type iterable:iterable
    :param predicate:
    :type predicate: (e)->Bool
    :return:
    :rtype: e | None
    """
    try:
        return next(i for i in iterable if predicate(i))
    except StopIteration:
        return None


def ilast(iter):
    """
    consume iterable and return last returned value
    """
    item = None
    for item in iter:
        pass
    return item
