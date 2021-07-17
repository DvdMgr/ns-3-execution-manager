from sem.utils import filter_logs, insert_logs, parse_logs, wipe_results
import sem
from sem import list_param_combinations, automatic_parser, stdout_automatic_parser, parse_log_components
import json
import numpy as np
import pytest
import os
import time
from operator import getitem

from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware


log_ls = ['+0.000000000s -1 [node -1] FrameExchangeManager:SetWifiMac(0x5576595683e0, 0x557659603820)\n',
          '+0.045510017s 1 [mac=00:00:00:00:00:01] FrameExchangeManager:RxStartIndication(0x5576595683e0, "PSDU reception started for ", +76us, " (txVector: ", txpwrlvl: 17 preamble: LONG channel width: 20 GI: 800 NTx: 97 Ness: 0 MPDU aggregation: 0 STBC: 0 FEC coding: BCC mode: OfdmRate6Mbps Nss: 1, ")")\n',
          '+0.000000000s -1 WifiPhy:SetChannelNumber(): [DEBUG] Saving channel number configuration for initialization \n',
          '+0.506390387s 1 [mac=00:00:00:00:00:01] FrameExchangeManager:StartTransmission(): [DEBUG] MPDU payload size=36, to=00:00:00:00:00:02, seq=16 \n',
          '+2.999074264s 0 [mac=00:00:00:00:00:02] FrameExchangeManager:NotifyReceivedNormalAck(0x56239764d8c0, DATA, payloadSize=1456, to=00:00:00:00:00:01, seqN=2323, duration/ID=+44000ns, lifetime=+195539us, packet=0x56239786dd00)\n',
          '+0.000000000s -1 WifiPhy:WifiPhy(0x557659571bd0)\n',
          '+0.000000000s -1 WifiPhy:SetChannelNumber(): [INFO  ] :()[]:./-=\n',
          '+2.999825333s 1 WifiPhy:GetTxPowerForTransmission(): [DEBUG ] txPowerDbm=17 with txPowerDbmPerMhz=3.9897 over 20 MHz\n',
          '+2.999825333s 1 WifiPhy:GetTxPowerForTransmission(): [INFO ] txPowerDbm=17 after applying m_powerDensityLimit=100\n'
          'WifiPhy:AddStaticPhyEntity(HR/DSSS) \n',
          '+0.000000000s -1 WifiPhy:SetChannelNumber): [DEBUG] Saving channel number configuration for initialization \n',
          '+0.000000000s -1 WifiPhy:SetChannelNumber(): DEBUG] Saving channel number configuration for initialization \n',
          '+0.000000000s WifiPhy:SetChannelNumber(): [DEBUG] Saving channel number configuration for initialization \n',
          '+0.000000000s -1 WifiPhy:SetChannelNumber(): Saving channel number configuration for initialization \n',
          '+0.000000000s -1 WifiPhy:SetChannelNumber():  \n',
          ]

expected_list = [{
                    'time': 0.000000000,
                    'context': '-1',
                    'extended_context': 'node -1',
                    'component': 'FrameExchangeManager',
                    'function': 'SetWifiMac',
                    'arguments': '0x5576595683e0, 0x557659603820',
                    'severity_class': 'FUNCTION',
                    'message': ''
                },
                {
                    'time': 0.045510017,
                    'context': '1',
                    'extended_context': 'mac=00:00:00:00:00:01',
                    'component': 'FrameExchangeManager',
                    'function': 'RxStartIndication',
                    'arguments': '0x5576595683e0, "PSDU reception started for ", +76us, " (txVector: ", txpwrlvl: 17 preamble: LONG channel width: 20 GI: 800 NTx: 97 Ness: 0 MPDU aggregation: 0 STBC: 0 FEC coding: BCC mode: OfdmRate6Mbps Nss: 1, ")"',
                    'severity_class': 'FUNCTION',
                    'message': ''
                },
                {
                    'time': 0.000000000,
                    'context': '-1',
                    'extended_context': None,
                    'component': 'WifiPhy',
                    'function': 'SetChannelNumber',
                    'arguments': '',
                    'severity_class': 'DEBUG',
                    'message': 'Saving channel number configuration for initialization'
                },
                {
                    'time': 0.506390387,
                    'context': '1',
                    'extended_context': 'mac=00:00:00:00:00:01',
                    'component': 'FrameExchangeManager',
                    'function': 'StartTransmission',
                    'arguments': '',
                    'severity_class': 'DEBUG',
                    'message': 'MPDU payload size=36, to=00:00:00:00:00:02, seq=16'
                },
                {
                     'time': 2.999074264,
                     'context': '0',
                     'extended_context': 'mac=00:00:00:00:00:02',
                     'component': 'FrameExchangeManager',
                     'function': 'NotifyReceivedNormalAck',
                     'arguments': '0x56239764d8c0, DATA, payloadSize=1456, to=00:00:00:00:00:01, seqN=2323, duration/ID=+44000ns, lifetime=+195539us, packet=0x56239786dd00',
                     'severity_class': 'FUNCTION',
                     'message': ''
                 },
                {
                    'time': 0.000000000,
                    'context': '-1',
                    'extended_context': None,
                    'component': 'WifiPhy',
                    'function': 'WifiPhy',
                    'arguments': '0x557659571bd0',
                    'severity_class': 'FUNCTION',
                    'message': ''
                },
                {
                    'time': 0.000000000,
                    'context': '-1',
                    'extended_context': None,
                    'component': 'WifiPhy',
                    'function': 'SetChannelNumber',
                    'arguments': '',
                    'severity_class': 'INFO',
                    'message': ':()[]:./-='
                },
                {
                     'time': 2.999825333,
                     'context': '1',
                     'extended_context': None,
                     'component': 'WifiPhy',
                     'function': 'GetTxPowerForTransmission',
                     'arguments': '',
                     'severity_class': 'DEBUG',
                     'message': 'txPowerDbm=17 with txPowerDbmPerMhz=3.9897 over 20 MHz'
                 },
                {
                     'time': 2.999825333,
                     'context': '1',
                     'extended_context': None,
                     'component': 'WifiPhy',
                     'function': 'GetTxPowerForTransmission',
                     'arguments': '',
                     'severity_class': 'INFO',
                     'message': 'txPowerDbm=17 after applying m_powerDensityLimit=100'
                 }]


def test_list_param_combinations():
    # Two possible combinations
    d1 = {'a': [1, 2]}
    assert set(map(lambda x: getitem(x, 'a'),
                   list_param_combinations(d1))) == set([1, 2])

    # Two possible combinations with two specified values
    d2 = {'b': [3], 'c': [4, 5]}
    assert set(map(lambda x: getitem(x, 'b'),
                   list_param_combinations(d2))) == set([3, 3])
    assert set(map(lambda x: getitem(x, 'c'),
                   list_param_combinations(d2))) == set([4, 5])

    # Four combinations with two specified values
    d3 = {'d': [6, 7], 'e': [8, 9]}
    assert sorted(map(lambda x: json.dumps(x, sort_keys=True),
                      list_param_combinations(d3))) == sorted(
                          map(lambda x: json.dumps(x, sort_keys=True),
                              [{'d': 6, 'e': 8}, {'d': 7, 'e': 8},
                               {'d': 6, 'e': 9}, {'d': 7, 'e': 9}]))

    params = {
        'a': [1],
        'b': [1, 2, 3],
    }
    params_more = {
        'a': [3],
        'b': [1, 2, 3],
    }
    params_same_as_above = {
        'a': [1, 3],
        'b': [1, 2, 3],
    }
    assert (sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       list_param_combinations(params_same_as_above)))
            ==
            sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       list_param_combinations([params, params_more]))))

    params_with_lambda = {
        'a': [1, 3],
        'b': lambda p: [10 * p['a']]
    }
    assert (sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       list_param_combinations(params_with_lambda)))
            ==
            sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       [{'a': 1, 'b': 10}, {'a': 3, 'b': 30}])))

    params_with_lambda_returning_list = {
        'a': [1, 3],
        'b': lambda p: list(range(p['a'])),
    }
    assert (sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       list_param_combinations(params_with_lambda_returning_list)))
            ==
            sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       [{'a': 1, 'b': 0}, {'a': 3, 'b': 0},
                        {'a': 3, 'b': 1}, {'a': 3, 'b': 2}])))

    params_with_two_lambdas = {
        'a': [1, 3],
        'b': lambda p: list(range(p['a'])),
        'c': lambda p: [10 * p['b']]
    }
    assert (sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       list_param_combinations(params_with_two_lambdas)))
            ==
            sorted(map(lambda x: json.dumps(x, sort_keys=True),
                       [{'a': 1, 'b': 0, 'c': 0}, {'a': 3, 'b': 0, 'c': 0},
                        {'a': 3, 'b': 1, 'c': 10}, {'a': 3, 'b': 2, 'c': 20}])))


def test_stdout_automatic_parser(result):
    # Create a dummy result
    result['output'] = {}
    result['output']['stderr'] = ''
    result['output']['stdout'] = '1 2 3 4 5\n6 7 8 9 10'

    parsed = stdout_automatic_parser(result)

    assert np.all(parsed == [[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10]])


def test_automatic_parser(result):
    # Create a dummy result
    result['output'] = {}
    result['output']['stderr'] = ''
    result['output']['stdout'] = '1 2 3 4 5\n6 7 8 9 10'

    parsed = automatic_parser(result)

    assert np.all(parsed['stdout'] == [[1, 2, 3, 4, 5],
                                       [6, 7, 8, 9, 10]])
    assert parsed['stderr'] == []


def test_parse_log_components(ns_3_compiled_debug, config):
    log_components = {
                'component1': 'info|prefix_level',
                'component2': 'level_debug|info',
                'component3': 'all|prefix_all',
                'component4': '**',
                'component5': '*|info',
                'component6': 'info|*',
                'component7': 'prefix_all'
            }

    new_components = parse_log_components(log_components=log_components)

    assert new_components == {
        'component1': 'info',
        'component2': 'error|warn|debug|info',
        'component3': 'error|warn|debug|info|function|logic',
        'component4': 'error|warn|debug|info|function|logic',
        'component5': 'error|warn|debug|info|function|logic',
        'component6': 'info',
        'component7': ''
    }

    log_components = {
        'component1': 'info|err'
    }

    with pytest.raises(ValueError):
        new_components = parse_log_components(log_components=log_components)

    new_campaign = sem.CampaignManager.new(ns_3_compiled_debug,
                                           'logging-ns3-script',
                                           config['campaign_dir'],
                                           overwrite=True,
                                           optimized=False)

    log_components = {
        '*': 'info',
        'Simulator': 'level_info'
    }

    with pytest.raises(ValueError):
        new_components = parse_log_components(log_components=log_components)

    ns3_log_components = new_campaign.runner.get_available_log_components()

    new_components = parse_log_components(log_components=log_components,
                                         ns3_log_components=ns3_log_components)

    assert new_components['Simulator'] == 'error|warn|debug|info'

    log_components = {
        'NonExistentLogComponent': 'info'
    }

    with pytest.raises(ValueError):
        new_components = parse_log_components(log_components=log_components,
                                             ns3_log_components=ns3_log_components)


def test_parse_logs():
    data_dir = os.path.join(os.curdir, 'tests', str(time.strftime("%Y-%m-%d::%H-%M-%S")) + ':test_logs.json')
    try:
        with open(data_dir, "w") as f:
            f.writelines(log_ls)

        parse_list = parse_logs(data_dir)
        assert len(parse_list) == len(expected_list)
        assert all([actual == expected for actual, expected in zip(parse_list, expected_list)])
    except RuntimeError:
        print('Test for parse_logs failed')
    finally:
        os.remove(data_dir)


def test_filters():
    db_dir = os.path.join(os.curdir, 'tests', str(time.strftime("%Y-%m-%d::%H-%M-%S")) + ':test_logs.json')
    data_dir = os.path.join(os.curdir, 'tests', str(time.strftime("%Y-%m-%d::%H-%M-%S")) + ':test_logs')

    try:
        db = TinyDB(db_dir,
                    storage=CachingMiddleware(JSONStorage))

        with open(data_dir, "w") as f:
            f.writelines(log_ls)

        parse_list = parse_logs(data_dir)
        insert_logs(parse_list, db)

        # Test case 1
        filter_list = filter_logs(db, context=['1'])
        expected_filter_list = [exp_list for exp_list in expected_list
                                if exp_list['context'] == '1']
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 2
        filter_list = filter_logs(db, context=['1'],
                                  function='GetTxPowerForTransmission')
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['context'] == '1' and
                                    exp_list['function'] == 'GetTxPowerForTransmission')]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 3
        filter_list = filter_logs(db, context=['1'], severity_class=['info'])
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['context'] == '1' and
                                    exp_list['severity_class'] == 'INFO')]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 4
        filter_list = filter_logs(db, context=['1'], time_begin=0.6)
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['context'] == '1' and
                                    exp_list['time'] >= 0.6)]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 5
        filter_list = filter_logs(db, context=['1'], time_end=0.5)
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['context'] == '1' and
                                    exp_list['time'] < 0.5)]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 6
        filter_list = filter_logs(db, severity_class='function')
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['severity_class'] == 'FUNCTION')]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 7
        filter_list = filter_logs(db, components={'WifiPhy': 'warn'})
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['component'] == 'WifiPhy' and
                                    exp_list['severity_class'] == 'WARN')]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 8
        filter_list = filter_logs(db, components={'WifiPhy': 'info'})
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['component'] == 'WifiPhy' and
                                    exp_list['severity_class'] == 'INFO')]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 9
        filter_list = filter_logs(db, components={'WifiPhy': ['info', 'debug']})
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['component'] == 'WifiPhy') and
                                   (exp_list['severity_class'] == 'INFO' or
                                    exp_list['severity_class'] == 'DEBUG')]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])

        # Test case 10
        filter_list = filter_logs(db,
                                  components={'WifiPhy': ['info', 'debug']},
                                  severity_class='function')
        expected_filter_list = [exp_list for exp_list in expected_list
                                if (exp_list['severity_class'] == 'FUNCTION') or
                                   ((exp_list['component'] == 'WifiPhy') and
                                    (exp_list['severity_class'] == 'INFO' or
                                     exp_list['severity_class'] == 'DEBUG'))]
        assert len(filter_list) == len(expected_filter_list)
        assert all([actual == expected for actual, expected
                    in zip(filter_list, expected_filter_list)])
    except RuntimeError:
        print('Test for filter logs failed')
    finally:
        wipe_results(db, db_dir)
        os.remove(data_dir)
