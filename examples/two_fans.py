
###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###


#from network import CooperativeNetwork, Retina, ExternalInputReader, SNNSimulation
from network.cooperative_net_change import *
from network.simulation import *
from network.ext_input_change import *
from network.retina_change import *
#from visualizer.visualizer import *
import os


def run_experiment_fans(with_visualization=True):
    """
    TODO: add experiment description.

    """
    experiment_name = "Back_Front"
    experiment_duration = 10000.0  # in ms
    dx = 92  # in pixels
    dy = 92  # in pixels
    max_d = 24  # in pixels
    crop_xmin = 25  # in pixels
    crop_ymin = 10  # in pixels

    # Setup the simulation
    Simulation = SNNSimulation(simulation_time=experiment_duration)
    ps.set_number_of_neurons_per_core(ps.IF_curr_exp, 255)
    ps.set_number_of_neurons_per_core(ps.SpikeSourceArray,92)

    # Define the input source
    path_to_input = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../data/input/Back_On_Front_Accel_Fixed_even.npz")
    ExternalRetinaInput = ExternalInputReader(file_path=path_to_input,
                                              dim_x=dx,
                                              dim_y=dy,
                                              crop_xmin=crop_xmin,
                                              crop_xmax=crop_xmin + dx,
                                              crop_ymin=crop_ymin,
                                              crop_ymax=crop_ymin + dy,
                                              sim_time=experiment_duration,
                                              is_rawdata_time_in_ms=False)
    # print(ExternalRetinaInput.retinaLeft)
    #print('it is not empty,right?')
    # Create two instances of Retinas with the respective inputs
    RetinaL = Retina(label="RetL", dimension_x=dx, dimension_y=dy,
                     spike_times=ExternalRetinaInput.retinaLeft,
                     record_spikes=True,
                     experiment_name=experiment_name)

    RetinaR = Retina(label="RetR", dimension_x=dx, dimension_y=dy,
                     spike_times=ExternalRetinaInput.retinaRight,
                     record_spikes=True,
                     experiment_name=experiment_name)

    # Create a cooperative network for stereo vision from retinal disparity

    SNN_Network = CooperativeNetwork(retinae={'left': RetinaL, 'right': RetinaR},
                                     max_disparity=max_d,
                                     record_spikes=True,
                                     record_v=False,
                                     experiment_name=experiment_name)

    print("hello,han,we are here, we are going to run the simulation")

    # Start the simulation
    Simulation.run()

    # Store the results in a file

    SNN_Network.get_spikes(sort_by_time=True, save_spikes=True)

    ret_right_spikes = RetinaR.get_spikes(sort_by_time=True, save_spikes=True)
    ret_left_spikes = RetinaL.get_spikes(sort_by_time=True, save_spikes=True)
    #print("the left spiketimes",ExternalRetinaInput.retinaLeft)
    #print("the right spiketimes",ExternalRetinaInput.retinaRight)

    # membrane_potential = SNN_Network.get_v(save_v=True)

    # Finish the simulation
    Simulation.end()
    print("boisoir,jai finis le simulation!")
    '''
    i = SNN_Network.i
    if with_visualization:
        from visualizer import Visualizer
        network_dimensions = SNN_Network.get_network_dimensions()

        viz = Visualizer(network_dimensions=network_dimensions,
                         experiment_name=experiment_name,
                         sim_time=experiment_duration,
                         spikes_file="./spikes/Back_Front_{0}_spikes.dat".format(i))
        # viz.microensemble_voltage_plot(save_figure=True)
        viz.disparity_histogram(over_time=False, save_figure=True)
        #viz.scatter_animation(dimension=2, save_animation=True, rotate=False)
    '''
