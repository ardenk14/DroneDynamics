import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import torch




if __name__ == '__main__':
    spacing = 0.1

    actions_lst = []
    states_lst = []
    for np_name in glob.glob('../*.np[yz]'):
        npz_data = np.load(np_name)
        states = npz_data['data'][:, :12]
        states_lst.append(states)
        actions = npz_data['data'][:, 12:]
        actions_lst.append(actions)
        print("STATES SHAPE: ", states.shape)
        print("ACTIONS SHAPE: ", actions.shape)

    actions_df = np.vstack(actions_lst)
    states_df = np.vstack(states_lst)
    
    print("ALL STATES: ", states_df.shape)
    print("ALL ACTIONS: ", actions_df.shape)

    actions_df = pd.DataFrame(actions_df, columns=["Deg3 Thrust", "Deg2 Thrust", "Deg1 Thrust", "Deg0 Thrust", "Deg3 Roll", "Deg2 Roll", "Deg1 Roll", "Deg0 Roll", "Deg3 Pitch", "Deg2 Pitch", "Deg1 Pitch", "Deg0 Pitch", "Deg3 Yaw", "Deg2 Yaw", "Deg1 Yaw", "Deg0 Yaw"])
    #print(actions_df.head)
    print(actions_df.describe())

    print(actions_df.cov())

    cov = actions_df.cov().to_numpy()
    cov[0:4, 0:4] *= 0.03
    #cov[3, 3] *= 0.000001
    cov[12:, :12] = 0.0
    cov[:12, 12:] = 0.0
    thrust_cov = torch.tensor([[1.06188844e+10, -1.52377202e+08,  5.37771266e+05, -1.84912403e+04],
                  [-1.52377202e+08,  2.32195539e+06, -1.09830688e+04,  2.25067188e+02],
                  [5.37771266e+05, -1.09830688e+04,  3.15472371e+02, -5.10406983e+00],
                  [-1.84912403e+04,  2.25067188e+02, -5.10406983e+00,  5.10962429e+01]])
    #thrust_cov[0, 0] /= 15
    #thrust_cov[-1, -1] *= 15
    #thrust_cov[-2, -2] *= 15
    roll_cov = torch.tensor(cov[4:8, 4:8])
    pitch_cov = torch.tensor(cov[8:12, 8:12])
    yaw_cov = torch.tensor(cov[12:, 12:])
    cov_new = torch.eye(16)
    cov_new[:4, :4] = thrust_cov
    cov_new[4:8, 4:8] = roll_cov
    cov_new[8:12, 8:12] = pitch_cov
    cov_new[12:, 12:] = yaw_cov

    cov_new[4, 4] *= 15
    cov_new[8, 8] *= 15
    cov_new[12, 12] *= 15
    cov_new[-1, -1] *= 15
    cov = cov_new
    #print("COV: ", cov_new)
    sample = np.random.multivariate_normal(actions_df.mean().to_numpy(), cov)
    print("SAMPLE: ", sample[:4])

    new_times = np.linspace(0, spacing, 20)
    #fitted_line_1 = np.round(np.polyval(sample[:4], new_times))

    print("Max: ", actions_df.max().to_numpy())
    print("Min: ", actions_df.min().to_numpy())
    print("MEAN: ", actions_df.mean().to_numpy())

    sample = np.zeros((50, 16))
    for i in range(50):
        sample[i] = np.random.multivariate_normal(actions_df.mean().to_numpy(), cov)

    print("Samples shape", sample.shape)

    new_times = np.linspace(0, spacing, 20)
    new_times = np.vander(new_times, N=4)
    print("NEW TIMES: ", new_times.shape)
    thrusts = sample[:, :4] @ new_times.T
    rolls = sample[:, 4:8] @ new_times.T
    pitches = sample[:, 8:12] @ new_times.T
    yaws = sample[:, 12:] @ new_times.T
    print("thrusts: ", thrusts.shape)

    #thrusts = np.round(np.polyval(sample[:, :4], new_times))
    #rolls = np.round(np.polyval(sample[:, 4:8], new_times))
    #pitches = np.round(np.polyval(sample[:, 8:12], new_times))
    #yaws = np.round(np.polyval(sample[:, 12:], new_times))
    act_vals = torch.tensor(np.hstack([thrusts, rolls, pitches, yaws]))
    #print("THRUSTS: ", thrusts.shape)
    print("ACT VALS: ", act_vals.shape)
    #print("VALUeS: ", (act_vals > 255).float())

    Q_bounds = torch.eye(act_vals.shape[-1]) * 8000
    print("Above: ", torch.diagonal((act_vals > 255).float() @ Q_bounds @ (act_vals > 255).float().t()))
    print("Below: ", torch.diagonal((act_vals < 0).float() @ Q_bounds @ (act_vals < 0).float().t()))

    cost = torch.diagonal((act_vals > 255).float() @ Q_bounds @ (act_vals > 255).float().t()) + torch.diagonal((act_vals < 0).float() @ Q_bounds @ (act_vals < 0).float().t())
    print("COSTS: ", cost)

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for i in range(50):
        if cost[i] > 0:
            print("Out of bounds")
            times = np.linspace(0, spacing, 20)
            #ax1.plot(times, act_vals[i, :20], c='orange', label='Fitted Line')
        else:
            times = np.linspace(0, spacing, 20)
            ax1.plot(times, act_vals[i, :20], c='g', label='Fitted Line')

    plt.show()
    plt.close()


    for j in range(10):
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        for i in range(100):
            new_times = np.linspace(0, spacing, 20)
            sample = np.random.multivariate_normal(actions_df.mean().to_numpy(), cov)
            fitted_line_1 = np.round(np.polyval(sample[:4], new_times))
            fitted_line_2 = np.round(np.polyval(sample[4:8], new_times))
            fitted_line_3 = np.round(np.polyval(sample[8:12], new_times))
            fitted_line_4 = np.round(np.polyval(sample[12:], new_times))

            ax1.plot(new_times, fitted_line_1, c='orange', label='Fitted Line')
            ax2.plot(new_times, fitted_line_2, c='orange', label='Fitted Line')
            ax3.plot(new_times, fitted_line_3, c='orange', label='Fitted Line')
            ax4.plot(new_times, fitted_line_4, c='orange', label='Fitted Line')

        #title = "Sequence " + str(i) + ", Step " + str(j)
        #fig1.suptitle(title)
        ax1.set_xlim(0, 0.1)
        ax1.set_ylim(0, 255)
        ax2.set_xlim(0, 0.1)
        ax2.set_ylim(0, 255)
        ax3.set_xlim(0, 0.1)
        ax3.set_ylim(0, 255)
        ax4.set_xlim(0, 0.1)
        ax4.set_ylim(0, 255)
        plt.show()

    # TODO: Sample potential action polynomials from multivariate normal distribution given by mean and cov
    # TODO: Ensure potential action polynomials make sense and don't go outside bounds of possible actions/training actions
    # TODO: If needed, see what happens when you divide the cov by certain factors

    #hist = actions_df.hist()
    plt.show()