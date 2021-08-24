


def get_models_info(clas):
    try:
        clas.__type__ = clas.__module__.split('.')[2].upper()
    except:
        clas.__type__ = ''
    try:
        clas.__task__ = clas.__module__.split('.')[3].replace('-', ' ')
    except:
        clas.__task__ = ''


