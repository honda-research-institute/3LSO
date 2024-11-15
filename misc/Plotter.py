import subprocess
import json
import seaborn as sns
from matplotlib.transforms import Affine2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../")

class Plotter:
    def __init__(self, logs, plot_config) -> None:
        self.state = np.array(logs["x"])
        self.obstacles = np.array(logs["xhat"])
        self.control = np.array(logs["u"])
        self.du = np.array(logs["du"])
        self.target = np.array(logs["target"])
        self.waypoints = logs["waypoints"]
        self.costs = np.array(logs["costs"])
        self.cost_star = np.array(logs["cost_star"])
        self.xhat_predictions = np.array(logs["xhat_predictions"])
        self.xhat_gt_predictions = np.array(logs["xhat_gt_predictions"])
        self.config = plot_config

    def plot_vehicles_gif(self, plot=False, label = 'output'):
        if plot:
            for k in range(self.obstacles.shape[0]):
                self.plot_vehicles(plot=True, k=k, savefig=True)

            command = [
                "ffmpeg",
                "-framerate",
                "20",
                "-i",
                "./images/vehicles_at_k%d.png",
                "-i",
                "./images/palette.png",
                "-lavfi",
                "paletteuse",
                f"./images/{label}.gif"
            ]
            subprocess.run(command)

    def plot_vehicles(self, plot=False, k=-1, savefig=False, label = 'vehicles_at_k'):
        if plot:
            plt.figure(figsize=(28, 4))
            xloc = self.state[k,0]
            yloc = self.state[k,1]
            
            # plt.xlim(0, 100)
            plt.xlim(xloc-20, xloc+50)
            plt.ylim(-5, 5)

            plt.grid(color="gray", linestyle="--")

            width = 5
            height = 2


            #plot ego vehicle
            ax = plt.gca()
            e_x, e_y = self.state[k, 0], self.state[k, 1]
            ax.add_patch(
                plt.Rectangle(
                    xy=(e_x - width / 2, e_y - height / 2),
                    width=width,
                    height=height,
                    edgecolor = 'k',
                    facecolor="g",
                    fill=True,
                    transform=Affine2D().rotate_deg_around(
                        *(e_x, e_y),
                        np.degrees(self.state[k, 2]),
                    )
                    + ax.transData,
                )
            )
            #plot obstacle vehicles
            for i in range(self.obstacles.shape[1]):
                for l in range(self.xhat_predictions.shape[2]):
                    a_x, a_y = (
                        self.xhat_predictions[k, i, l, 0],
                        self.xhat_predictions[k, i, l, 1],
                    )
                    ax.add_patch(
                        Rectangle(
                            xy=(a_x - width / 2, a_y - height / 2),
                            width=width,
                            height=height,
                            facecolor="mediumblue",
                            alpha=0.1,
                            transform=Affine2D().rotate_deg_around(
                            *(a_x - width / 2, a_y - height / 2),
                            np.degrees(self.obstacles[k, i, 2]),
                            ) + ax.transData,
                        )
                    )

                a_x, a_y = self.obstacles[k, i, 0], self.obstacles[k, i, 1]
                ax.add_patch(
                    Rectangle(
                        xy=(a_x - width / 2, a_y - height / 2),
                        width=width,
                        height=height,
                        edgecolor = 'k',
                        facecolor="mediumblue",
                        fill=True,
                        transform=Affine2D().rotate_deg_around(
                        *(a_x - width / 2, a_y - height / 2),
                        np.degrees(self.obstacles[k, i, 2]),
                        ) + ax.transData,
                    )
                )
                # plot predictions
                
            
            # plot target
            plt.scatter(
                self.target[k, 0], self.target[k, 1], c="red", linewidths=5, marker="*"
            )
            plt.scatter(
                np.array(self.waypoints[k])[:, :, 0].flatten(),
                np.array(self.waypoints[k])[:, :, 1].flatten(),
                c="k",
                marker=".",
                s=0.2
            )
            
            # plot history
            plt.plot(self.state[:k, 0], self.state[:k, 1], "g--")

            if savefig:
                plt.savefig(f"./images/{label}{k}.png")
                plt.close()
            else:
                plt.show()

    def plot_cost(self, plot=False, k=0, savefig=False):
        if plot:
            j_min = self.config["j_min"]  # [m/s3]
            j_max = self.config["j_max"]  # [m/s3]
            sr_min = self.config["sr_min"]  # [rad/s]
            sr_max = self.config["sr_max"]  # [rad/s]
            # size of jerk space, keep it an even number, Default 5
            N_j = self.config["N_j"]
            # size of steering rate space, keep it an even number, Default 5
            N_sr = self.config["N_sr"]

            # Generate action spaces
            j_space = np.linspace(j_min, j_max, N_j)
            sr_space = np.linspace(sr_min, sr_max, N_sr)

            annot_kws = {
                "fontsize": 5,
                "fontstyle": "italic",
                "color": "k",
                "alpha": 0.6,
                "verticalalignment": "center",
            }

            to_vis = np.resize(self.costs[k], (N_j, N_sr))
            sns.heatmap(
                to_vis,
                linewidth=0.5,
                xticklabels=j_space,
                yticklabels=sr_space,
                annot=True,
                annot_kws=annot_kws,
                fmt=".2f",
            )
            if savefig:
                plt.savefig(f"./images/costs-{k}.png")
                plt.close()
            else:
                plt.show()

    def plot_cost_star(self, plot=False, savefig=False, label = 'cost'):
        if plot:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
            
            axs[0].plot(np.arange(0,self.cost_star.shape[0],1), self.cost_star, 'b-', label='cost')
            axs[0].set_xlabel('Cost at each step')
            axs[0].grid()
            axs[0].legend()

            axs[1].plot(np.arange(0,self.cost_star.shape[0],1), np.cumsum(self.cost_star), 'r-', label=f'cumulative sum = {np.cumsum(self.cost_star)[-1]}')
            axs[1].set_xlabel('Cumulative cost')
            axs[1].grid()
            axs[1].legend()

            plt.tight_layout()
            
            if savefig:
                plt.savefig(f"./images/{label}.png")
                plt.close()
            else:
                plt.show()

    def plot_u(self, plot=False, savefig=False, label = 'control-inputs'):
        if plot:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
            
            # First subplot for control[:, 0]
            axs[0].plot(np.arange(0,self.state.shape[0],1), self.control[:, 0], 'b-', label='a')
            axs[0].plot(np.arange(1,self.state.shape[0],1), self.du[:, 0], 'b--', label=r'$\Delta$a')
            axs[0].set_ylabel('m/s^2')
            axs[0].grid()
            axs[0].legend()

            # Second subplot for control[:, 1]
            axs[1].plot(np.arange(0,self.state.shape[0],1), self.control[:, 1], 'r-', label=r'$\phi$')
            axs[1].plot(np.arange(1,self.state.shape[0],1), self.du[:, 1], 'r--', label=r'$\Delta\phi$')
            axs[1].set_xlabel('Timesteps')
            axs[1].set_ylabel('rad')
            axs[1].grid()
            axs[1].legend()

            plt.tight_layout()
            
            if savefig:
                plt.savefig(f"./images/{label}.png")
                plt.close()
            else:
                plt.show()

    def plot_model_accuracy(self, plot=False, savefig = False, label='model-accuracy'):
        if plot:
            error = np.sqrt((self.xhat_predictions - self.xhat_gt_predictions)**2)[1::10]
            mean = np.mean(error,axis=1)
            std = np.std(error,axis=1)
            
            #plot only for three time steps
            fig, axs = plt.subplots(nrows = 4, ncols = 1, sharex=True, figsize=(7,15))
            for tstep in range(error.shape[0]):
                # First subplot for x
                axs[0].plot(np.arange(0,error.shape[2],1), mean[tstep,:,0], label=f't={tstep*10}')
                axs[0].fill_between(np.arange(0,error.shape[2],1), mean[tstep,:,0]-std[tstep,:,0],mean[tstep,:,0]+std[tstep,:,0], alpha=0.2)
                axs[0].legend()
                axs[0].grid()

                # Second subplot for y
                axs[1].plot(np.arange(0,error.shape[2],1), mean[tstep,:,1], label='x')
                axs[1].fill_between(np.arange(0,error.shape[2],1), mean[tstep,:,1]-std[tstep,:,1],mean[tstep,:,1]+std[tstep,:,1], alpha=0.2)
                axs[1].grid()


                # Second subplot for psi
                axs[2].plot(np.arange(0,error.shape[2],1), mean[tstep,:,2], label='x')
                axs[2].fill_between(np.arange(0,error.shape[2],1), mean[tstep,:,2]-std[tstep,:,2],mean[tstep,:,2]+std[tstep,:,2], alpha=0.2)
                axs[2].grid()
                # Second subplot for v
                axs[3].plot(np.arange(0,error.shape[2],1), mean[tstep,:,3], label='x')
                axs[3].fill_between(np.arange(0,error.shape[2],1), mean[tstep,:,3]-std[tstep,:,3],mean[tstep,:,3]+std[tstep,:,3], alpha=0.2)
                axs[3].grid()

                axs[0].set_ylabel(r'x')
                axs[1].set_ylabel(r'y')
                axs[2].set_ylabel(r'$\psi$')
                axs[3].set_ylabel(r'$v$')
                axs[3].set_xlabel(f'predictive time steps')
                
            plt.tight_layout()
            
            if savefig:
                plt.savefig(f"./images/{label}.png")
                plt.close()
            else:
                plt.show()
