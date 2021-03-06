# plugins.py

import functools
import importlib
from collections import namedtuple
from importlib import resources

# Basic structure for storing information about one plugin
Plugin = namedtuple("Plugin", ("name", "func"))

# Dictionary with information about all registered plugins
_PLUGINS = {}


def register(clas):
    """Decorator for registering a new plugin"""
    package, _, _ = clas.__module__.rpartition(".")
    if getattr(clas, 'variants', None):
        variants = clas.variants
        for name in variants:
            _PLUGINS[name] = Plugin(name=name, func=functools.partial(clas, name))
    else:
        name = clas.__name__
        _PLUGINS[name] = Plugin(name=name, func=clas)
    return clas


def names(package):
    """List all plugins in one package"""
    _import_all(package)
    return sorted(_PLUGINS)


def get(package, plugin):
    """Get a given plugin"""
    plugins = [_p.split(package+'.')[1] for _p in load_plugins(package)]
    for _plugin in plugins:
        _import(package, _plugin)
    return _PLUGINS[plugin].func


def call(package, plugin, *args, **kwargs):
    """Call the given plugin"""
    plugin_func = get(package, plugin)
    return plugin_func(*args, **kwargs)


def _import(package, plugin):
    """Import the given plugin file from a package"""
    importlib.import_module(f"{package}.{plugin}")


def load_plugins(package):
    files = list(resources.contents(package))
    plugins = [f'{package}.{f[:-3]}' for f in files if f.endswith(".py") and f[0] != "_"]
    for _sub_dir in files:
        if _sub_dir != 'saved_models':
            if (_sub_dir[0] != '_') & (not _sub_dir.endswith('.py')):
                plugins.extend(load_plugins(f'{package}.{_sub_dir}'))
    return plugins


def _import_all(package):
    """Import all plugins in a package"""
    plugins = [_p.split(package+'.')[1] for _p in load_plugins(package)]
    for plugin in plugins:
        _import(package, plugin)


def names_factory(package):
    """Create a names() function for one package"""
    return functools.partial(names, package)


def get_factory(package):
    """Create a get() function for one package"""
    return functools.partial(get, package)


def call_factory(package):
    """Create a call() function for one package"""
    return functools.partial(call, package)


