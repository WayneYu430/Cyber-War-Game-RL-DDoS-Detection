import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from scipy.interpolate import make_interp_spline


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed
"""
attacker_reached = cbw_logger.attacker_reached
user_reached = cbw_logger.user_reached
attacker_rew = cbw_logger.attack_reward
util = cbw_logger.util
defend_reward = cbw_logger.defend_reward
user_reached_p = cbw_logger.user_reached_p
attacker_reached_p = cbw_logger.attack_reached_p
a_done = cbw_logger.a_done
d_done = cbw_logger.d_done

"""


def draw(logger, step_per_epoch, step_per_collect, save=False, name=''):
    attacker_reached = logger.attacker_reached
    user_reached = logger.user_reached
    attacker_rew = logger.attack_reward
    util = logger.util
    defend_reward = logger.defend_reward

    a_done = logger.a_done
    d_done = logger.d_done

    # print(attacker_rew[1])
    rew_array_all = []
    rew_array_defend = []
    user_reached_all = []
    attacker_reached_all = []
    user_reached_p = []
    attacker_reached_p = []
    user_reached_p_all = []
    attacker_reached_p_all = []

    user_reached_p = logger.user_reached_p
    attacker_reached_p = logger.attack_reached_p

    a_max_rew = 0
    b_max_rew = 0
    user_max_reached = 0
    # 1. print reward graphs
    fig1, ax = plt.subplots(figsize=(12,8))
    # ax.plot(attacker_reached[::2], attacker_rew[::2], label='attacker')
    fig2, ax2 = plt.subplots(figsize=(16,8))
    fig3, ax3 = plt.subplots(figsize=(12,8))
    fig4, ax4 = plt.subplots(figsize=(12,8))
    fig5, ax5 = plt.subplots(figsize=(16,8))
    fig6, ax6 = plt.subplots(figsize=(16,8))
    fig7, ax7 = plt.subplots(figsize=(16,8))
    fig8, ax8 = plt.subplots(figsize=(16,8))

    for i in range(len(attacker_rew)):
        if i <= 0:
            continue
        a_max_rew = max(np.array(attacker_rew[i]))
        b_max_rew = max(np.array(defend_reward[i]))
        user_max_reached = max(np.array(user_reached[i]))
        attack_max_reached = max(np.array(user_reached[i]))
        reached_max = max(attack_max_reached,user_max_reached)
    step=0
    for i in range(len(attacker_rew)):
        if i <= 0:
            continue
        step += len(attacker_rew[i])
        for j in range(len(attacker_rew[i])):
            ax2.vlines(step, 0, a_max_rew, linestyle=(0, (3, 10, 1, 10)), colors='0.5')
            ax3.vlines(step, 0, 1, linestyle=(0, (3, 10, 1, 10)), colors='0.5')
            ax5.vlines(step, 0, b_max_rew, linestyle=(0, (3, 10, 1, 10)), colors='0.5')
            ax6.vlines(step, 0, a_max_rew, linestyle=(0, (3, 10, 1, 10)), colors='.5')
            ax7.vlines(step, 0, reached_max, linestyle=(0, (3, 10, 1, 10)), colors='.5')
            ax8.vlines(step, 0.5, 1, linestyle=(0, (3, 10, 1, 10)), colors='.5')
            # a_max_rew = max(attacker_rew[i][j], a_max_rew)
            rew_array_all.append(float(attacker_rew[i][j]))
            rew_array_defend.append(float(defend_reward[i][j]))

    print("a {}, b {}".format(a_max_rew, b_max_rew))
    # print(rew_array_defend)

    # 2. set title and axis
    ax.set_xlabel('Attacker Packets Reached')
    ax.set_ylabel('Attacker Reward')
    ax.set_title('Attacker & Reward ')

    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Reward (Epochs start at 0)')
    ax2.set_title('Attacker Reward Learning Curve')

    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Utilization of the Server')
    ax3.set_title('Server Utilization Monitor')

    ax4.set_xlabel('Epochs Done')
    ax4.set_ylabel('Steps')
    ax4.set_title('Learning Steps')

    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Defender Reward Learning Curve')
    ax5.set_title('Defend Rewards')

    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Attacker VS Defender Reward Learning Curve')
    ax6.set_title('Rewards')

    ax7.set_xlabel('Steps')
    ax7.set_ylabel('Packets Reached')
    ax7.set_title('Attacker VS User Packets Reached The Server')

    ax8.set_xlabel('Steps')
    ax8.set_ylabel('Packets Reached Probability')
    ax8.set_title('Attacker VS User Packets Probability Reached The Server')


    step_take = []
    # 3. Draw the reward for each epochs
    for i in range(len(attacker_rew)):
        # ax.plot(attacker_reached[i], attacker_rew[i], label='attacker')
        # ax2.plot(attacker_rew[i], label='attacker')
        # Util smooth
        """
        if i>=len(attacker_rew)-5:
            util_x = np.array(range(len(util[i])))
            util_y = util[i]
            X_Y_Spline = make_interp_spline(util_x, util_y)
            X_ = np.linspace(util_x.min(), util_x.max(), 100)
            Y_ = X_Y_Spline(X_)
            ax3.plot(util_x, util_y)
        """

        if i>=1:
            util_x = np.array(range(len(util[i])))
            util_y = np.array(util[i])
            util_y_sm = smooth(util_y, 0.9)
            current_step = 0
            for i_step in step_take:
                current_step+=int(i_step)
            ax3.plot(current_step+util_x, util_y_sm)
            # if i % 10 == 0 or i == 1:
            ax3.annotate('Epoch-'+str(i), xy=(current_step+util_x[-1], util_y_sm[-1]),
                     xytext=(current_step+util_x[-1], util_y_sm[-1]), arrowprops=dict(facecolor='black', shrink=0.05))
            # ax3.plot(util_x, util_y)
            step_take.append(len(attacker_rew[i]))
            # print(len(attacker_reached[i]), attacker_reached[i])
            for itemU in user_reached[i]:
                user_reached_all.append(float(itemU))
            for itemA in attacker_reached[i]:
                attacker_reached_all.append(float(itemA))
            for up in user_reached_p[i]:
                if up != 0.0:
                    user_reached_p_all.append(round(float(up), 2))
            for ap in attacker_reached_p[i]:
                if ap != 0.0:
                    attacker_reached_p_all.append(round(float(ap), 2))
            # user_reached_all.append(np.asarray(user_reached[i], dtype=float))
            # attacker_reached_all.append(np.asarray(attacker_reached[i], dtype=float))

    # ax.plot(attacker_reached, attacker_rew, label='defender')
    # ax.plot(user_reached, defend_reward, label='defender')
    # 2. print utilization graph
    ax2.plot(rew_array_all, linestyle='--', label='attacker')
    # ax2.plot([0]*len(rew_array_all), label='bottom')
    # ax3.plot([0.4]*max(len(i) for i in attacker_rew), linestyle='solid', label='threshold')
    ax4.plot(step_take, label='Steps Attacker Take')
    ax5.plot(rew_array_defend, linestyle='-', label='defender')
    ax6.plot(rew_array_defend, linestyle='-', label='defender')
    ax6.plot(rew_array_all, label='attacker')
    ax7.plot(range(len(user_reached_all)), smooth(user_reached_all, 0.9), label='user')
    ax7.plot(range(len(attacker_reached_all)), smooth(attacker_reached_all, 0.9), label='attacker')

    ax8.plot(range(len(user_reached_p_all)), smooth(user_reached_p_all, 0.9), label='user')
    ax8.plot(range(len(attacker_reached_p_all)), smooth(attacker_reached_p_all, 0.9), label='attacker')


    # ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    ax8.legend()

    # Save Figure in file with format in Time-
    def save_train(step_per_epoch=step_per_epoch, step_per_collect=step_per_collect):
        # train_args = get_args()
        step_per_epoch = step_per_epoch
        step_per_collect = step_per_collect
        now_time = datetime.datetime.now()

        script_dir = os.path.dirname('train_results_RandomDefender/')
        results_dir = os.path.join(script_dir, now_time.strftime("%m-%d %H-%M")+'-epoch_step-' +\
                                   str(step_per_epoch) + ' collect_step-' + str(step_per_collect) \
                                   +'-'+name+'/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        fig1.savefig(os.path.join(results_dir, 'attacker_reward_Packets.png'))
        fig2.savefig(os.path.join(results_dir, 'attacker_reward.png'))
        fig3.savefig(os.path.join(results_dir, 'Server Utilization.png'))
        fig4.savefig(os.path.join(results_dir, 'Attacker Learning_step.png'))
        fig5.savefig(os.path.join(results_dir, 'defender_reward.png'))
        fig6.savefig(os.path.join(results_dir, 'Defender VS Attacker.png'))
        fig7.savefig(os.path.join(results_dir, 'Attacker & User Reached.png'))
        fig8.savefig(os.path.join(results_dir, 'Attacker & User Reached In Prob.png'))

    print("Attacker wins {}, And defender wins {}".format(a_done[-1], d_done[-1]))
    plt.show()
    if save:
        save_train()
