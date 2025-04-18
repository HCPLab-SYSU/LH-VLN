import numpy as np
import json
import math
import shutil
import os
import time
from habitat_base.simulation import SceneSimulator
from dataset import TaskDataset, count_gt_length


# eval for one multi nav task
def eval_for_one_task(args, config):
    task_sim = SceneSimulator(args=args, config=config)
    print(config)
    config['Geo dis'] = []
    config['trial'] = {}
    # config_map(task_sim.sim, task_sim.save_path)

    # init
    success = 0
    action = 'stop'
    obs = task_sim.actor(action, -1, success)

    nav_step = []
    obj = []        # finished target object
    
    for i in range(len(task_sim.target)):
        _, _, _, _, k = task_sim.get_info(i)
        if math.isinf(k):
            return config['Task instruction']
        if i == 0:
            config['Geo dis'].append(k)

    # nav loop
    for step in range(args.max_step): 
        # get the current target and geo_dis
        obj_target = task_sim.target[success]
        coord_list = task_sim.get_coord(obj_target)
        if not(coord_list):
            return config['Task instruction']
        snap_coord_list = [task_sim.pathfinder.snap_point(coord) for coord in coord_list]
        geo_dis, coord = task_sim.geodesic_distance(snap_coord_list)

        if math.isinf(geo_dis):
            return config['Task instruction']

        # record the trial
        key = 'trial_' + str(success)
        pos, yaw = task_sim.return_state()
        if key not in config['trial']:
            config['trial'][key] = {}
            config['trial'][key]['pos'] = []
            config['trial'][key]['yaw'] = []
            config['trial'][key]['action'] = []
            config['trial'][key]['pos'].append(pos.tolist())
            config['trial'][key]['yaw'].append(yaw)
            config['trial'][key]['action'].append(action)
        else:
            config['trial'][key]['pos'].append(pos.tolist())
            config['trial'][key]['yaw'].append(yaw)
            config['trial'][key]['action'].append(action)
    

        # termination
        if geo_dis < args.success_dis:  # success
            obj.append(obj_target)  # add finished target
            success = success + 1

            # record the trial
            action = 'stop'

            config['trial'][key]['pos'].append(pos.tolist())
            config['trial'][key]['yaw'].append(yaw)
            config['trial'][key]['action'].append(action)
            obs = task_sim.actor(action, step-1, success-1)

            print("\n***** nav to %s success! *****\n" % obj_target)
            if len(nav_step) == 0:
                nav_step.append(step)
            else:
                former = nav_step[-1]
                nav_step.append(step-former)
            
            _, _, _, _, geo_dis = task_sim.get_info(success)
            config['Geo dis'].append(geo_dis)
            
        if success == len(task_sim.target):
            break
        
        # get the current target
        obj_target, coord, position, yaw, geo_dis = task_sim.get_info(success)

        if math.isinf(geo_dis):
            return config['Task instruction']

        action = task_sim.get_next_action(coord)
        obs = task_sim.actor(action, step, success)

    # save data
    save_path = task_sim.save_path
    # plot the top down map
    # config_map(task_sim.sim, save_path)
    task_sim.sim.close()
    if success == len(task_sim.target):     # success trial
        print("\n\n***** Task finished !! *****")
        for i in range(success):
            print("***** nav to %s *****" % task_sim.target[i])
        temp_path = save_path + '/temp'
        if not os.path.isdir(temp_path):
            print("file wasn't saved!")
            return [True, sum(nav_step), nav_step]
        success_path = save_path + '/success'
        if not os.path.isdir(success_path):
            os.mkdir(success_path)
        exist = len(os.listdir(success_path)) + 1
        shutil.move(temp_path, success_path + '/trial_' + str(exist))
        with open(success_path + '/trial_' + str(exist) + '/task.json', 'w') as json_file:
            json.dump(config, json_file, indent=4)
        print("success record!")
        return [True, sum(nav_step), nav_step]
        
    else:                                   # failure trial
        print("\n\n***** Reach the max step !! *****")
        for i in range(success):
            print("***** nav to %s *****" % task_sim.target[i])
        print("***** %d target left *****" % (len(task_sim.target)-success))
        temp_path = save_path + '/temp'
        if not os.path.isdir(temp_path):
            print("file wasn't saved!")
            return [False, None, nav_step]
        fail_path = save_path + '/fail'
        if not os.path.isdir(fail_path):
            os.mkdir(fail_path)
        exist = len(os.listdir(fail_path)) + 1
        bi = str(success) + '-' + str(len(task_sim.target))
        shutil.move(temp_path, fail_path + '/trial_' + str(exist) + '_' + bi)
        with open(fail_path + '/trial_' + str(exist) + '_' + bi + '/task.json', 'w') as json_file:
            json.dump(config, json_file, indent=4)
        print("failure record!")
        return [False, None, nav_step]


def gen_traj(args):
    dataset = TaskDataset(args)
    # category = {2:{}, 3:{}, 4:{}}
    # for task in dataset:
    #     if task["Robot"] not in category[len(task['Object'])]:
    #         category[len(task['Object'])][task['Robot']] = 1
    #     else:
    #         category[len(task['Object'])][task['Robot']] += 1
    # with open('category/batch_5_category.json', 'w') as json_file:
    #     json.dump(category, json_file, indent=4)
    # print(len(dataset)) 85
    # task steps: step num for each task
    # nav steps: step num for each single navigation
    # success: num of success tasks
    # nav success: num of success navigation
    # nav total: num of total navigation
    task_steps = []
    nav_steps = []
    success = 0
    nav_success = 0
    nav_total = 0
    times = []
    fail_tasks = []
    # len(dataset)
    # print(len(dataset))
    # for i in range(0, 100):
    #     print(i)
    #     print(dataset[i]['Task instruction'])
    
    start = 0
    end = start + len(dataset)
    for i in range(start, end):
        config = dataset[i]
        time_start = time.time()
        print("***** %d *****" % i)
        out = eval_for_one_task(args, config)        
        # continue
        print("***** %d *****" % i)
        if type(out) == str:
            fail_tasks.append(out)
            print("***** The target of task %s is un reachable! *****" % out)
            continue
        time_end = time.time()
        if out[0]:
            success += 1
            task_steps.append(out[1])
            nav_steps.extend(out[2])
            nav_success += len(out[2])
            times.append((time_end - time_start)/sum(out[2]))
        else:
            nav_steps.extend(out[2])
            nav_success += len(out[2])
            times.append((time_end - time_start)/(1000*args.max_step))
        
        nav_total += len(config['Object'])

    total_task = end - start - len(fail_tasks)
    print("***** mean task step %f *****" % (np.mean(task_steps)))
    print("***** mean nav step %f *****" % (np.mean(nav_steps)))
    print("***** task success: %f, task total: %f *****" % (success, total_task))
    print("***** nav success: %f, nav total: %f *****" % (nav_success, nav_total))
    print("***** mean task SR %f *****" % (success/total_task if total_task != 0 else 0))
    print("***** mean nav SR %f *****" % (nav_success/nav_total if nav_total != 0 else 0))

    result = {
        'time cost': times,
        'fail tasks': fail_tasks,
        'mean task step': np.mean(task_steps),
        'mean nav step': np.mean(nav_steps),
        'mean task SR': success/total_task if total_task != 0 else 0,
        'mean nav SR': nav_success/nav_total if nav_total != 0 else 0,
    }
    with open('logs/' + str(start) + '_' + str(end) + '_result.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)
