# -*- coding: utf-8 -*-


from ..utils import plugins

models_names = plugins.names_factory(__package__)
load = plugins.call_factory(__package__)

