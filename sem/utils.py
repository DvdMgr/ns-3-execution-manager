import io
import math
import copy
import warnings
import re
import os
import time
from copy import deepcopy
from pprint import pformat
from operator import and_, or_
from pathlib import Path
from itertools import product
from functools import reduce, wraps

import matplotlib.pyplot as plt
import numpy as np
import numpy.core.numeric as nx
import SALib.analyze.sobol
import SALib.sample.saltelli

from tinydb import TinyDB, where, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

try:
    DRMAA_AVAILABLE = True
    import drmaa
except(RuntimeError):
    DRMAA_AVAILABLE = False


def output_labels(argument):
    def decorator(function):
        function.__dict__["output_labels"] = argument
        @wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator


def only_load_some_files(argument):
    def decorator(function):
        function.__dict__["files_to_load"] = argument
        @wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator


def yields_multiple_results(function):
    function.__dict__["yields_multiple_results"] = True
    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        return result
    return wrapper


def list_param_combinations(param_ranges):
    """
    Create a list of all parameter combinations from a dictionary specifying
    desired parameter values as lists.

    Example:

        >>> param_ranges = {'a': [1], 'b': [2, 3]}
        >>> list_param_combinations(param_ranges)
        [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

    Additionally, this function is robust in case values are not lists:

        >>> param_ranges = {'a': 1, 'b': [2, 3]}
        >>> list_param_combinations(param_ranges)
        [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

    """
    param_ranges_copy = copy.deepcopy(param_ranges)
    # If we are passed a list, we want to expand each nested specification
    if isinstance(param_ranges_copy, list):
        return sum([list_param_combinations(x) for x in param_ranges_copy], [])
    # If it's a dictionary, we need to make sure lists with 1 item are reduced
    # to the item itself.
    if isinstance(param_ranges_copy, dict):
        # Convert non-list values to single-element lists
        for key, value in param_ranges_copy.items():
            if isinstance(value, list) and len(value) == 1:
                param_ranges_copy[key] = value[0]
    # If it's a dictionary and all items are lists, we need to expand it
    if isinstance(param_ranges_copy, dict):
        for key, value in param_ranges_copy.items():
            if isinstance(value, list):
                # Expand all values that are not functions
                new_dictionaries = []
                for v in value:
                    c = copy.deepcopy(param_ranges_copy)
                    c[key] = [v]
                    new_dictionaries += [c]
                # Iterate again to check
                return list_param_combinations(new_dictionaries)
    # If we get to this point, we have a dictionary and all items have length 1
    # Now it's time to expand the functions.
    if isinstance(param_ranges_copy, dict):
        for key, value in param_ranges_copy.items():
            if callable(value):
                param_ranges_copy[key] = value(param_ranges_copy)
                return list_param_combinations(param_ranges_copy)
    return [param_ranges_copy]


def get_command_from_result(script, result, debug=False):
    """
    Return the command that is needed to obtain a certain result.

    Args:
        params (dict): Dictionary containing parameter: value pairs.
        debug (bool): Whether the command should include the debugging
            template.
    """
    if not debug:
        command = "python3 waf --run \"" + script + " " + " ".join(
            ['--%s=%s' % (param, value) for param, value in
             result['params'].items()]) + "\""
    else:
        command = "python3 waf --run " + script + " --command-template=\"" +\
            "gdb --args %s " + " ".join(['--%s=%s' % (param, value) for
                                         param, value in
                                         result['params'].items()]) + "\""
    return command


def constant_array_parser(result):
    """
    Dummy parser, used for testing purposes.
    """
    return [0, 1, 2, 3]


def automatic_parser(result, dtypes={}, converters={}):
    """
    Try and automatically convert strings formatted as tables into nested
    list structures.

    Under the hood, this function essentially applies the genfromtxt function
    to all files in the output, and passes it the additional kwargs.

    Args:
      result (dict): the result to parse.
      dtypes (dict): a dictionary containing the dtype specification to perform
        parsing for each available filename. See the numpy genfromtxt
        documentation for more details on how to format these.
    """
    np.seterr(all='raise')
    parsed = {}

    # By default, if dtype is None, the order Numpy tries to convert a string
    # to a value is: bool, int, float. We don't like this, since it would give
    # us a mixture of integers and doubles in the output, if any integers
    # existed in the data. So, we modify the StringMapper's default mapper to
    # skip the int check and directly convert numbers to floats.
    oldmapper = np.lib._iotools.StringConverter._mapper
    np.lib._iotools.StringConverter._mapper = [(nx.bool_,
                                                np.lib._iotools.str2bool,
                                                False),
                                               (nx.floating, float, nx.nan),
                                               (nx.complexfloating, complex,
                                                nx.nan + 0j),
                                               (nx.longdouble, nx.longdouble,
                                                nx.nan)]

    for filename, contents in result['output'].items():
        if dtypes.get(filename) is None:
            dtypes[filename] = None
        if converters.get(filename) is None:
            converters[filename] = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed[filename] = np.genfromtxt(io.StringIO(contents),
                                             dtype=dtypes[filename],
                                             converters=converters[filename]
                                             ).tolist()

    # Here we restore the original mapper, so no side-effects remain.
    np.lib._iotools.StringConverter._mapper = oldmapper

    return parsed


def stdout_automatic_parser(result):
    """
    Try and automatically convert strings formatted as tables into a matrix.

    Under the hood, this function essentially applies the genfromtxt function
    to the stdout.

    Args:
      result (dict): the result to parse.
    """
    np.seterr(all='raise')
    parsed = {}

    # By default, if dtype is None, the order Numpy tries to convert a string
    # to a value is: bool, int, float. We don't like this, since it would give
    # us a mixture of integers and doubles in the output, if any integers
    # existed in the data. So, we modify the StringMapper's default mapper to
    # skip the int check and directly convert numbers to floats.
    oldmapper = np.lib._iotools.StringConverter._mapper
    np.lib._iotools.StringConverter._mapper = [(nx.bool_,
                                                np.lib._iotools.str2bool,
                                                False),
                                               (nx.floating, float, nx.nan),
                                               (nx.complexfloating, complex,
                                                nx.nan + 0j),
                                               (nx.longdouble, nx.longdouble,
                                                nx.nan)]

    file_contents = result['output']['stdout']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = np.genfromtxt(io.StringIO(file_contents))

    # Here we restore the original mapper, so no side-effects remain.
    np.lib._iotools.StringConverter._mapper = oldmapper

    return parsed


#################################
# Code for sensitivity analysis #
#################################


def get_bounds(ranges):
    """
    Format bounds for SALib, starting from a dictionary of ranges for each
    parameter. The values for the parameters contained in ranges can be one of
    the following:
    1. A dictionary containing min and max keys, describing a range of possible
    values for the parameter.
    2. A list of allowed values for the parameter.
    """
    bounds = {}
    for i in ranges.items():
        if isinstance(i[1], dict):
            # Defined as range
            bounds[i[0]] = [i[1]['min'], i[1]['max']]
        elif len(i[1]) > 1:
            # Defined as list of possible values
            bounds[i[0]] = [0, len(i[1])]

    return bounds


def salib_param_values_to_params(ranges, values):
    """
    Convert SALib's parameter specification to a SEM-compatible parameter
    specification.
    """
    sem_params = []
    for value in values:
        v_idx = 0
        params = {}
        for rang in ranges.items():
            if isinstance(rang[1], dict):
                # Defined as range, leave as it is
                params[rang[0]] = value[v_idx]
                v_idx += 1
            elif len(rang[1]) > 1:
                # Defined as list of possible values
                params[rang[0]] = ranges[rang[0]][math.floor(value[v_idx])]
                v_idx += 1
            else:
                # Defined as single value
                params[rang[0]] = rang[1][0]
        sem_params.append(params)
    return sem_params


def compute_sensitivity_analysis(
        campaign, result_parsing_function, ranges,
        salib_sample_function=SALib.sample.saltelli.sample,
        salib_analyze_function=SALib.analyze.sobol.analyze,
        samples=100):
    """
    Compute sensitivity analysis on a campaign using the passed SALib sample
    and analyze functions.
    """
    bounds = get_bounds(ranges)

    problem = {
        'num_vars': len(bounds),
        'names': list(bounds.keys()),
        'bounds': list(bounds.values())}
    param_values = salib_sample_function(problem, samples)
    sem_parameter_list = salib_param_values_to_params(ranges, param_values)

    if not bounds.get('RngRun'):
        # If we don't have RngRun parameter specified, we just assign a new
        # value to each combination
        next_runs = campaign.db.get_next_rngruns()
        for p in sem_parameter_list:
            p['RngRun'] = next(next_runs)

    # TODO Make a copy of all available results, search a result for each item
    # in sem_parameter_list, remove the result from the copied list, assign new
    # RngRun value in case we don't find anything.

    campaign.run_missing_simulations(sem_parameter_list)
    results = np.array(
        [result_parsing_function(campaign.db.get_complete_results(p)[0])
         for p in sem_parameter_list])
    return salib_analyze_function(problem, results)


# def interactive_plot(campaign, param_ranges, result_parsing_function, x_axis,
#                      runs=None):
#     # Average over RngRuns if param_ranges does not contain RngRun
#     if runs is not None:
#         assert(not param_ranges.get('RngRun'))
#         xarray = campaign.get_results_as_xarray(param_ranges,
#                                                 result_parsing_function,
#                                                 'Result',
#                                                 runs).reduce(np.mean, 'runs')
#     else:
#         assert(param_ranges.get('RngRun'))
#         xarray = campaign.get_results_as_xarray(param_ranges,
#                                                 result_parsing_function,
#                                                 'Result',
#                                                 runs=1)

#     def plot_line(**kwargs):
#         # x goes on the x axis
#         # Everything else goes as a parameter
#         # plt.xlabel(x_axis)
#         plt.ylim([np.min(xarray), np.max(xarray)])
#         plt.plot(param_ranges[x_axis],
#                  np.array(xarray.sel(**kwargs)).squeeze())
#     interact(plot_line, **{k: v for k, v in param_ranges.items() if k != x_axis
#                            and len(v) > 1})

def parse_log_components(log_components, ns3_log_components=None):
    """
    Verifies if the log levels/log classes passed in the log_component
    dictionary are valid and converts log levels to corresponding log classes.
    Returns a dictionary with the valid components and log classes.
    For example,
    'level_debug' gets converted to 'warn|error|debug'

    Note: If log_components is None or {}(Empty dictionary), None is returned.

    Args:
        log_components (dict): a python dictionary with the
            log_components (to enable) as the key and the log_levels
            as the value. Log levels should be written in the same format as
            the one specified by the ns-3 manual for the environment variable
            NS_LOG.
            Note: If any prefix is mentioned, it will be dropped and
            prefix_all will always be appended.

            For example,
            log_components = {
                'component1' : 'info',
                'component2' : 'level_debug|info'
            }
        ns3_log_components (list): A list containing all the valid log
            components supported by ns-3.
    """
    if not log_components:
        return None

    log_components_dict = {}
    log_level_list = ['error',
                      'warn',
                      'debug',
                      'info',
                      'function',
                      'logic']
    converter = {
        'error': ['error'],
        'warn': ['warn'],
        'debug': ['debug'],
        'info': ['info'],
        'function': ['function'],
        'logic': ['logic'],
        'all': ['error', 'warn', 'debug', 'info', 'function', 'logic'],
        'level_error': ['error'],
        'level_warn': ['error', 'warn'],
        'level_debug': ['error', 'warn', 'debug'],
        'level_info': ['error', 'warn', 'debug', 'info'],
        'level_function': ['error', 'warn', 'debug', 'info', 'function'],
        'level_logic': ['error', 'warn', 'debug', 'info', 'function', 'logic'],
        'level_all': ['error', 'warn', 'debug', 'info', 'function', 'logic'],
        '**': ['error', 'warn', 'debug', 'info', 'function', 'logic'],
        '*': None,
        'prefix_func': None,
        'prefix_time': None,
        'prefix_node': None,
        'prefix_level': None,
        'prefix_all': None
    }
    for component, levels in log_components.items():
        log_level_complete = set()
        if (ns3_log_components is not None) and component not in ns3_log_components and component != '*':
            raise ValueError(
                "Log component '%s' is not a valid ns-3 log component. Valid log components: \n%ls" % (
                    component,
                    ns3_log_components
                ))
        for level in levels.split('|'):
            if level not in converter:
                raise ValueError("Log level '%s' for component '%s' is not valid"
                                 % (level,
                                    component))

            # '*' represents level_all only if it occurs before the first '|'
            if levels.split('|')[0] == '*':
                log_level_complete.update(converter['all'])
                continue

            # Do not update the dictionary if prefixes are mentioned
            if converter[level] is not None:
                log_level_complete.update(converter[level])

        # Update log_level_complete if entry for components exists
        if component in log_components_dict:
            log_level_complete.update(log_components_dict[component].split('|'))

        # Sort the log classes for consistency
        log_level_sorted = [level for level in log_level_list
                            if level in log_level_complete]

        if component == '*':
            if ns3_log_components is None:
                raise ValueError('No list of ns-3 supported log components passed.\n')
            for comp in ns3_log_components:
                log_components_dict[comp] = "|".join(log_level_sorted)
        else:
            log_components_dict[component] = "|".join(log_level_sorted)

    return log_components_dict


def convert_environment_str_to_dict(log_components):
    """
    Converts NS_LOG formatted string to a dictionary.

    For example,
        log_components =
            'NS_LOG="component1=info:component2=level_debug|info"'
    will be converted to
        dict = {
            'component1': 'info',
            'component2': 'level_debug|info'
        }

    Args:
        log_components (str): a string formatted in the NS_LOG variable
            format.
    """
    # Droppping 'NS_LOG=' prefix and check if the format of the provided
    # string is valid. The validation of log components and log levels is
    # done in Utils.parse_log_components.
    # TODO - Use split() instead of regex
    groups = re.match(r'^NS_LOG="((?:(\*\*\*)|([a-zA-Z]+)|(?:\*|[a-zA-Z]+)=(?:[a-zA-Z_]+|\*|\*\*)(\|(?:[a-zA-Z_]+|\*|(\*\*)))*)(:(?:(\*\*\*)|([a-zA-Z]+)|(?:\*|[a-zA-Z]+)=(?:[a-zA-Z_]+|\*|\*\*)(\|(?:[a-zA-Z_]+|\*|(\*\*)))*))*)"$', log_components)

    if groups is None:
        raise ValueError("Provided log_components string '%s' is invalid\n" % log_components)

    # convert the string to log_components dictionary format
    # to store in the database later
    log_components_dict = {}
    for component in groups.group(1).split(':'):
        component_and_level = component.split('=')
        if len(component_and_level) == 1:
            if component_and_level[0] == '***':
                log_components_dict['*'] = 'all'
                return log_components_dict
            else:
                log_components_dict[component_and_level[0]] = 'all'

        elif len(component_and_level) == 2:
            log_components_dict[component_and_level[0]] = component_and_level[1]

        else:
            raise ValueError("Provided log_components string '%s' is invalid\n" % log_components)

    return log_components_dict


def process_logs(log_file):
    """
    Create a TinyDB instance, parse the log file and insert the logs to a
    TinyDB instance.

    Returns a TinyDB instance containing the logs from the log file and the
    path to where that instance is stored.

    Args:
        log_file (str): Path to where the log file is stored
    """
    if not Path(log_file).exists():
        raise FileNotFoundError("Cannot access file '%s'\n" % log_file)

    logs = parse_logs(log_file)

    data_dir = os.path.join('/tmp/', str(time.strftime("%Y-%m-%d::%H-%M-%S")) + ':logs.json')
    if Path(data_dir).exists():
        raise FileExistsError("File path '%s' already exists" % data_dir)

    db = TinyDB(data_dir,
                storage=CachingMiddleware(JSONStorage))

    insert_logs(logs, db)
    # TODO Find a more elegant way to clean up TinyDB directory after
    # completion
    return db, data_dir


def parse_logs(log_file):
    """
    Parse the logs from a log file.

    Return a list of dictionaries with each dictionary having the following
    format:
    dictionary = {
        'time': timestamp,  # float
        'context': context/nodeId,  # str
        'Extended_Context': ,   #str
        'component': log component,  # str
        'function': function name,  # str
        'arguments': function arguments,  # str
        'severity_class': log severity class,  # str
        'message': log message  # str
    }
    Note: This function will skip the log lines that do not have the same
          structure as ns-3 logs with prefix level set to prefix_all.

    Args:
        log_file (string): Path to where the log file is stored
    """
    log_list = []

    # Regex for parsing the logs
    # time: Accepts string of form "[+-]'digits.digits's". The number of digits
    #       is variable as user can change time resolution.
    time_re = r'[\+\-]?(?P<time>\d+\.\d+)s '
    # context: Accepts string of form "digits" or "-digits".
    # extended_context: Accepts everything between [] followed by a space. i.e.
    #                   '[Extended Context] '.
    # Note: '.*?' will try to match minimum possible string. Or in other words,
    #       it is a non greedy match.
    context_extended_context_re = r'(?P<context>(?:\d+|-\d+)) (?:\[(?P<extended_context>.*?)\] )?'
    # component: Accepts string of form '[a-zA-Z0-9_]'
    # function: Accepts string of form '[a-zA-Z0-9_]'
    # arguments: Accepts everything between '()'(after function name).
    # Note: '.*?' will try to match minimum possible string. Or in other words,
    #       it is a non greedy match.
    component_function_arguments_re = r'(?P<component>\w+):(?P<function>\w+)\((?P<arguments>.*?)\)'
    # severity_class: Accepts string of form '[a-zA-Z0-9_]'
    # message: Accepts everything after 'severity_class '.
    # Note: '\s*' matches extra spaces if present after severity_class.
    # Note: 'severity_class_message_re' is optional as when
    #       severity_class=function severity_class and message are not present
    #       in the resultant log. 
    severity_class_message_re = r'(?:: \[(?P<severity_class>\w+)\s*\] (?P<message>.*))?'

    regex = re.compile(r'^' + time_re + context_extended_context_re + component_function_arguments_re + severity_class_message_re + r'$')

    with open(log_file) as f:
        for log in f:
            # Groups structure
            # group[1] = Time
            # group[2] = Context
            # group[3] = Extended Context; For example, '-1 [node -1]' group[2] = -1 and group[3] = node -1
            # group[4] = Component
            # group[5] = Function
            # group[6] = Arguments
            # group[7] = Severity_class
            # group[8] = Message

            # Example: '+0.000000000s -1 PowerAdaptationDistance:SetupPhy(): [DEBUG] OfdmRate6Mbps 0.00192 6000000bps'
            # group[1] = 0.000000000
            # group[2] = -1
            # group[3] = None
            # group[4] = PowerAdaptationDistance
            # group[5] = SetupPhy
            # group[6] = ''
            # group[7] = DEBUG
            # group[8] = OfdmRate6Mbps 0.00192 6000000bps
            groups = regex.match(log)

            if groups is None:
                warnings.warn("Log format is not consistent with prefix_all. Skipping log '%s'" % log, RuntimeWarning)
                continue

            temp_dict = None
            # Remove trailing whitespaces after message
            message = groups.group('message')
            if message is not None:
                message = message.rstrip()

            # If level is function
            if groups.group('severity_class') is None and groups.group('message') is None:
                temp_dict = {
                    'time': float(groups.group('time')),
                    'context': groups.group('context'),
                    'extended_context': groups.group('extended_context'),
                    'component': groups.group('component'),
                    'function': groups.group('function'),
                    'arguments': groups.group('arguments'),
                    'severity_class': 'FUNCTION',
                    'message': ''
                }
            else:
                temp_dict = {
                    'time': float(groups.group('time')),
                    'context': groups.group('context'),
                    'extended_context': groups.group('extended_context'),
                    'component': groups.group('component'),
                    'function': groups.group('function'),
                    'arguments': groups.group('arguments'),
                    'severity_class': groups.group('severity_class'),
                    'message': message
                }
            log_list.append(temp_dict)

    return log_list


def insert_logs(logs, db):
    """
    Insert the logs in the TinyDB instance passed.

    Note: This function does not return anything.

    Args:
        logs (list): A list of logs to insert in database. Logs are described
                     as a python dict
        db (TinyDB instance): A TinyDB instace where the logs will be inserted.
    """
    if logs == [] or logs is None:
        return

    example_result = {
        k: ['...'] for k in ['time',
                             'context',
                             'component',
                             'extended_context',
                             'function',
                             'arguments',
                             'severity_class',
                             'message']
    }

    for log in logs:
        # Verify log format is correct
        # Only check if the keys are consistent
        if not(set(log.keys()) == set(example_result.keys())):
            raise ValueError(
                '%s:\nExpected: %s\nGot: %s' % (
                    "Log dictionary does not correspond to database format",
                    pformat(example_result, depth=2),
                    pformat(log, depth=2)))

    db.table('logs').insert_multiple(deepcopy(logs))


def wipe_results(db, data_dir):
    """
    Remove all logs from the database.

    This also removes all output files, and cannot be undone.

    Note: This function does not return anything.

    Args:
        db (TinyDB instance): A TinyDB instace where the logs are inserted.
        data_dir (str): Path to where the TinyDB instance is stored
    """
    # Clean logs table
    db.drop_table('logs')
    db.storage.flush()

    # Get rid of contents of data dir
    try:
        os.remove(data_dir)
    except OSError as error:
        print(error)
        print("File path '%s' can not be removed" % data_dir)


def filter_logs(db,
                severity_class=None,
                components=None,
                function=None,
                context=None,
                time_begin=None,
                time_end=None):
    """
    Filter the logs stored in the database.

    Filters are applied on context, function name, log sevirity class and time.
    Additionally the user can also filter each log component based on a
    particular sevirity class using components dictionary.
    For example, if the user specifies Context = [0, 1] and Function = [A, B]
    the function will output logs in which (context == 0 or context == 1) and
    (function == a or function == b).

    Return a list of logs that satisfy all the passed filters. Each log is
    represented by a dictionary.

    Args:
        db (TinyDB instance): A TinyDB instace where the logs are inserted.
        context (list, str, int): A list of context based on which the logs
            will be filtered. If only one context is to be provided, then this
            paramter can also be a string or an int.
        function (list, str): A list of function names based on which the logs
            will be filtered. If only one function name is to be provided, then
            this paramter can also be a string.
        time_begin (float): Start timestamp (in seconds) of the time window.
        time_end (float): End timestamp (in seconds) of the time window.
        severity_class (list, str): A list of log severity classes based on
            which the logs will be filtered. If only one log severity class is
            to be provided, then this paramter can also be a string.
        components (dict): A dictionary having structure
            {
                components:['class1','class2']
            }
            based on which the logs will be filtered.
    """
    query_final = []

    # Assert that the passed paramters are of valid type.
    if severity_class is not None:
        if severity_class is not None:
            if isinstance(severity_class, str):
                severity_class = [severity_class]
            elif isinstance(severity_class, list):
                pass
            else:
                raise TypeError("severity_class can only be a list or a string (if only one value is passed).")

    if components is not None:
        for key, value in components.items():
            if isinstance(value, str):
                components[key] = [value]
            elif isinstance(value, list):
                pass
            else:
                raise TypeError("values in components dictionary can only be a list or a string (if only one value is passed).")

    if function is not None:
        if isinstance(function, str):
            function = [function]
        elif isinstance(function, list):
            pass
        else:
            raise TypeError("function can only be a list or a string (if only one value is passed).")

    if context is not None:
        if isinstance(context, str) or isinstance(context, int):
            context = [str(context)]
        elif isinstance(context, list):
            pass
        else:
            raise TypeError("context can only be a list or a string (if only one value is passed).")

    # Build TinyDB query based on the passed paramters
    if severity_class is not None or components is not None:
        query_list = []
        if severity_class is not None:
            query = reduce(or_, [
                           where('severity_class') == lvl.upper()
                           for lvl in severity_class
                           ])
            query_list.append(query)
        # If components is provided apply the specified log severity classes
        # to the specified log components in addition to the log severity
        # classes passed with 'severity_class'. In other words, log severity
        # classes passed with 'severity_class' is treated as a global filter.
        if components is not None:
            query = reduce(or_, [reduce(or_, [
                    Query().fragment({'component': component,
                                      'severity_class': cls.upper()})
                    for cls in classes])
                    for component, classes in components.items()])
            query_list.append(query)

        query_final.append(reduce(or_, query_list))

    if function is not None:
        query = reduce(or_, [where('function') == fnc for fnc in function])
        query_final.append(query)

    if context is not None:
        query = reduce(or_, [where('context') == str(ctx) for ctx in context])
        query_final.append(query)

    if time_begin is not None:
        query = where('time') >= float(time_begin)
        query_final.append(query)

    if time_end is not None:
        query = where('time') <= float(time_end)
        query_final.append(query)

    if query_final is not None:
        query = reduce(and_, query_final)
        return [dict(i) for i in db.table('logs').search(query)]
    else:
        return []
