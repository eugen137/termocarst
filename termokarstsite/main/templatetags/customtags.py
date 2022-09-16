from django import template


register = template.Library()


def index(indexable, i):
    return indexable[int(i)]


def range_index(indexable):
    return range(len(indexable))


def get_json(indexable):
    return None
    s = list(indexable.values_list('value'))
    print(s)
    d = []
    for q in s:
        d.append(list(q))
    return d


register.filter('index', index)
register.filter('get_json', get_json)
register.filter('range_index', range_index)
