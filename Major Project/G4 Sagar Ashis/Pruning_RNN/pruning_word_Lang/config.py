import logging
from collections import OrderedDict
import yaml
import json
from torch.optim.lr_scheduler import *
import utils
from scheduler import CompressionScheduler
from policy import PruningPolicy
from pruning import *
import utils

msglogger = logging.getLogger()
app_cfg_logger = logging.getLogger("app_cfg")


def dict_config(model, optimizer, sched_dict, scheduler=None, resumed_epoch=None):
    app_cfg_logger.debug('Schedule contents:\n' + json.dumps(sched_dict, indent=2))

    if scheduler is None:
        scheduler = CompressionScheduler(model)

    pruners = __factory('pruners', model, sched_dict)
    regularizers = __factory('regularizers', model, sched_dict)
    quantizers = __factory('quantizers', model, sched_dict, optimizer=optimizer)
    if len(quantizers) > 1:
        raise ValueError("\nError: Multiple Quantizers not supported")
    extensions = __factory('extensions', model, sched_dict)

    try:
        lr_policies = []
        for policy_def in sched_dict['policies']:
            policy = None
            if 'pruner' in policy_def:
                try:
                    instance_name, args = __policy_params(policy_def, 'pruner')
                except TypeError as e:
                    print('\n\nFatal Error: a policy is defined with a null pruner')
                    print('Here\'s the policy definition for your reference:\n{}'.format(json.dumps(policy_def, indent=1)))
                    raise
                assert instance_name in pruners, "Pruner {} was not defined in the list of pruners".format(instance_name)
                pruner = pruners[instance_name]
                policy = PruningPolicy(pruner, args)
            add_policy_to_scheduler(policy, policy_def, scheduler)

        lr_schedulers = __factory('lr_schedulers', model, sched_dict, optimizer=optimizer,
                                  last_epoch=(resumed_epoch if resumed_epoch is not None else -1))
        for policy_def in lr_policies:
            instance_name, args = __policy_params(policy_def, 'lr_scheduler')
            assert instance_name in lr_schedulers, "LR-scheduler {} was not defined in the list of lr-schedulers".format(
                instance_name)
            lr_scheduler = lr_schedulers[instance_name]
            policy = distiller.LRPolicy(lr_scheduler)
            add_policy_to_scheduler(policy, policy_def, scheduler)

    except AssertionError:
        raise
    except Exception as exception:
        print("\nFATAL Parsing error!\n%s" % json.dumps(policy_def, indent=1))
        print("Exception: %s %s" % (type(exception), exception))
        raise
    return scheduler


def add_policy_to_scheduler(policy, policy_def, scheduler):
    if 'epochs' in policy_def:
        scheduler.add_policy(policy, epochs=policy_def['epochs'])
    else:
        scheduler.add_policy(policy, starting_epoch=policy_def['starting_epoch'],
                            ending_epoch=policy_def['ending_epoch'],
                            frequency=policy_def['frequency'])


def file_config(model, optimizer, filename, scheduler=None, resumed_epoch=None):
    """Read the schedule from file"""
    with open(filename, 'r') as stream:
        msglogger.info('Reading compression schedule from: %s', filename)
        try:
            sched_dict = utils.yaml_ordered_load(stream)
            return dict_config(model, optimizer, sched_dict, scheduler, resumed_epoch)
        except yaml.YAMLError as exc:
            print("\nFATAL parsing error while parsing the schedule configuration file %s" % filename)
            raise


def config_component_from_file_by_class(model, filename, class_name, **extra_args):
    with open(filename, 'r') as stream:
        msglogger.info('Reading configuration from: %s', filename)
        try:
            config_dict = distiller.utils.yaml_ordered_load(stream)
            config_dict.pop('policies', None)
            for section_name, components in config_dict.items():
                for component_name, user_args in components.items():
                    if user_args['class'] == class_name:
                        msglogger.info(
                            'Found component of class {0}: Name: {1} ; Section: {2}'.format(class_name, component_name,
                                                                                            section_name))
                        user_args.update(extra_args)
                        return build_component(model, component_name, user_args)
            raise ValueError(
                'Component of class {0} does not exist in configuration file {1}'.format(class_name, filename))
        except yaml.YAMLError:
            print("\nFATAL parsing error while parsing the configuration file %s" % filename)
            raise


def __factory(container_type, model, sched_dict, **extra_args):
    container = {}
    if container_type in sched_dict:
        for name, user_args in sched_dict[container_type].items():
            try:
                instance = build_component(model, name, user_args, **extra_args)
                container[name] = instance
            except Exception as exception:
                print("\nFatal error while parsing [section: %s] [item: %s]" % (container_type, name))
                print("Exception: %s %s" % (type(exception), exception))
                raise

    return container


def build_component(model, name, user_args, **extra_args):
    class_name = user_args.pop('class')
    try:
        class_ = globals()[class_name]
    except KeyError as ex:
        raise ValueError("Class named '{0}' does not exist".format(class_name)) from ex

    valid_args, invalid_args = utils.filter_kwargs(user_args, class_.__init__)
    if invalid_args:
        raise ValueError(
            '{0} does not accept the following arguments: {1}'.format(class_name, list(invalid_args.keys())))

    valid_args.update(extra_args)
    valid_args['model'] = model
    valid_args['name'] = name
    final_valid_args, _ = utils.filter_kwargs(valid_args, class_.__init__)
    instance = class_(**final_valid_args)
    return instance


def __policy_params(policy_def, type):
    name = policy_def[type]['instance_name']
    args = policy_def[type].get('args', None)
    return name, args