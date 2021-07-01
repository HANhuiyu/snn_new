#import spynnaker.pyNN as ps
import pyNN.spiNNaker as ps
import numpy as np
import time
import os


class CooperativeNetwork(object):

    def __init__(self, retinae=None,
                 max_disparity=0, cell_params=None,
                 record_spikes=True, record_v=False, experiment_name="Experiment",
                 verbose=True):
        # IMPORTANT NOTE: This implementation assumes min_disparity = 0

        assert retinae['left'] is not None and retinae['right'] is not None, \
            "ERROR: Retinas are not initialised! Creating Network Failed."

        dx = retinae['left'].dim_x
        assert dx > max_disparity >= 0, "ERROR: Maximum Disparity Constant is illegal!"
        self.max_disparity = max_disparity
        self.min_disparity = 0
        self.size = (2 * (dx - self.min_disparity) * (self.max_disparity - self.min_disparity + 1)
                     - (self.max_disparity - self.min_disparity + 1) ** 2
                     + self.max_disparity - self.min_disparity + 1) / 2
        self.dim_x = dx
        self.dim_y = retinae['left'].dim_y

        # check this assertion before the actual network generation, since the former
        # might take very long to complete.
        assert retinae['left'].dim_x == retinae['right'].dim_x and \
            retinae['left'].dim_y == retinae['right'].dim_y, \
            "ERROR: Left and Right retina dimensions are not matching. Connecting Spike Sources to Network Failed."

        # TODO: make parameter values dependent on the simulation time step
        # (for the case 0.1 it is not tested completely and should serve more like an example)

        # the notation for the synaptic parameters is as follows:
        # B blocker, C collector, S spike source, (2, 4)
        # w weight, d delay, (1)
        # a one's own, z other, (3)
        # i inhibition, e excitation  (5)
        # If B is before C than the connection is from B to C.
        # Example: dSaB would mean a dealy from a spike source to the one's own blocker neuron, and
        # wSzB would be the weight from a spike source to the heterolateral blocker neuron.
        params = {'neural': dict(), 'synaptic': dict(), 'topological': dict()}
        simulation_time_step = 0.2
        if simulation_time_step == 0.2:
            params['neural'] = {'tau_E': 2.0,
                                'tau_I': 2.0,
                                'tau_mem': 2.07,
                                'v_reset_blocker': -84.0,
                                'v_reset_collector': -90.0}  # why -90.0?
            w = 18.0
            params['synaptic'] = {'wBC': w,  # -20.5: negative won't work. However keep in mind that it is inhibitory!
                                  'dBC': simulation_time_step,
                                  'wSC': w,
                                  'dSC': 1.6,
                                  'wSaB': w,
                                  'dSaB': simulation_time_step,
                                  'wSzB': w,    # same story here
                                  'dSzB': simulation_time_step,
                                  'wCCi': w,    # and again
                                  'dCCi': simulation_time_step,
                                  'wCCo': w/3,  # and again
                                  'dCCo': simulation_time_step,
                                  'wCCe': 1.8,
                                  'dCCe': simulation_time_step}
            params['topological'] = {'radius_e': 1,
                                     'radius_i': max(self.dim_x, self.dim_y)}
        elif simulation_time_step == 0.1:
            params['neural'] = {'tau_E': 1.0,
                                'tau_I': 1.0,
                                'tau_mem': 1.07,
                                'v_reset_blocker': -92.0,
                                'v_reset_collector': -102.0}
            params['synaptic'] = {'wBC': 39.5,  # weight should be positive numbers, indicated as inhibitory synapses (if necessary)!
                                  'dBC': simulation_time_step,
                                  'wSC': 39.5,
                                  'dSC': 0.8,
                                  'wSaB': 49.5,
                                  'dSaB': simulation_time_step,
                                  'wSzB': 39.5,  # same here
                                  'dSzB': simulation_time_step,
                                  'wCCi': 50.0,  # and here
                                  'dCCi': simulation_time_step,
                                  'wCCe': 4.0,
                                  'dCCe': simulation_time_step}
            params['topological'] = {'radius_e': 1,
                                     'radius_i': max(self.dim_x, self.dim_y)}

        self.cell_params = params if cell_params is None else cell_params

        self.network = self._create_network(record_spikes=record_spikes,
                                            record_v=record_v,
                                            verbose=verbose)

        self._connect_spike_sources(retinae=retinae, verbose=verbose)

        self.experiment_name = experiment_name.replace(" ", "_")

    def _create_network(self, record_spikes=False, record_v=False, verbose=False):

        print("INFO: Creating Cooperative Network of size {0} (in microensembles).".format(
            self.size))

#        if record_spikes:
#            from pyNN.spiNNaker import record

        network = []
        neural_params = self.cell_params['neural']
        for y in range(0, self.dim_y):
            blockerLeft = ps.Population(self.size,
                                        ps.IF_curr_exp,
                                        {'tau_syn_E': neural_params['tau_E'],
                                         'tau_syn_I': neural_params['tau_I'],
                                         'tau_m': neural_params['tau_mem'],
                                         'v_reset': neural_params['v_reset_blocker']},
                                        label="Blocker_left{0}".format(y))

            blockeRight = ps.Population(self.size,
                                        ps.IF_curr_exp,
                                        {'tau_syn_E': neural_params['tau_E'],
                                         'tau_syn_I': neural_params['tau_I'],
                                         'tau_m': neural_params['tau_mem'],
                                         'v_reset': neural_params['v_reset_blocker']},
                                        label="Blocker_right{0}".format(y))
            collector = ps.Population(self.size,
                                      ps.IF_curr_exp,
                                      {'tau_syn_E': neural_params['tau_E'],
                                       'tau_syn_I': neural_params['tau_I'],
                                       'tau_m': neural_params['tau_mem'],
                                       'v_reset': neural_params['v_reset_collector']},
                                      label="Collector {0}".format(y))
            if record_spikes:
                collector.record('spikes')  # records only the spikes
            if record_v:
                collector.record_v()  # records the membrane potential -- very resource demanding!
                blocker.record_v()

            network.append((blockerLeft, blockeRight, collector))

        self._interconnect_neurons(network, verbose=verbose)

        if self.dim_x > 1:
            self._interconnect_neurons_inhexc(network, verbose)
        else:
            global _retina_proj_l, _retina_proj_r, same_disparity_indices
            _retina_proj_l = [[0]]
            _retina_proj_r = [[0]]
            same_disparity_indices = [[0]]

        return network

    def _interconnect_neurons(self, network, verbose=False):

        assert network is not None, \
            "ERROR: Network is not initialised! Interconnecting failed."

        synaptic_params = self.cell_params['synaptic']

        # generate connectivity list: 0 untill dimensionRetinaY-1 for the left
        # and dimensionRetinaY till dimensionRetinaY*2 - 1 for the right
        # the connection bloker with collector, here change the

        # connect the inhibitory neurons to the cell output neurons
        if verbose:
            print("INFO: Interconnecting Neurons. This may take a while.")

        for ensemble in network:
            ps.Projection(ensemble[0], ensemble[2], ps.OneToOneConnector(),
                          ps.StaticSynapse(weight=synaptic_params['wBC'],
                                           delay=synaptic_params['dBC']),
                          receptor_type='inhibitory')
            ps.Projection(ensemble[1], ensemble[2], ps.OneToOneConnector(),
                          ps.StaticSynapse(weight=synaptic_params['wBC'],
                                           delay=synaptic_params['dBC']),
                          receptor_type='inhibitory')

    def _interconnect_neurons_inhexc(self, network, verbose=False):

        assert network is not None, \
            "ERROR: Network is not initialised! Interconnecting for inhibitory and excitatory patterns failed."

        if verbose and self.cell_params['topological']['radius_i'] < self.dim_x:
            print(
                "WARNING: Bad radius of inhibition. Uniquness constraint cannot be satisfied.")
        if verbose and 0 <= self.cell_params['topological']['radius_e'] > self.dim_x:
            print("WARNING: Bad radius of excitation. ")

        # create lists with inhibitory along the Retina Right projective line
        nbhoodInhL = []
        nbhoodInhR = []
        nbhoodExcX = []

        # used for the triangular form of the matrix in order to remain within the square
        if verbose:
            print("INFO: Generating inhibitory and excitatory connectivity patterns.")
        # generate rows
        limiter = self.max_disparity - self.min_disparity + 1
        ensembleIndex = 0

        while ensembleIndex < self.size:
            if ensembleIndex / (self.max_disparity - self.min_disparity + 1) > \
                    (self.dim_x - self.min_disparity) - (self.max_disparity - self.min_disparity) - 1:
                limiter -= 1
                if limiter == 0:
                    break
            nbhoodInhL.append(
                [ensembleIndex + disp for disp in range(0, limiter)])
            ensembleIndex += limiter

        ensembleIndex = self.size

        # generate columns
        nbhoodInhR = [[x] for x in nbhoodInhL[0]]
        shiftGlob = 0
        for x in nbhoodInhL[1:]:
            shiftGlob += 1
            shift = 0

            for e in x:
                if (shift + 1) % (self.max_disparity - self.min_disparity + 1) == 0:
                    nbhoodInhR.append([e])
                else:
                    nbhoodInhR[shift + shiftGlob].append(e)
                shift += 1

        # generate all diagonals
        '''
        for diag in map(None, *nbhoodInhL):
            sublist = []
            for elem in diag:
                if elem is not None:
                    sublist.append(elem)
            nbhoodExcX.append(sublist)
	'''
        length = max(map(len, nbhoodInhL))
        arr = np.array([xi+[None]*(length-len(xi)) for xi in nbhoodInhL])
        nbhoodExcX = []
        for i in range(0, length):
            rowlist = arr[:, i].tolist()
            sublist = list(x for x in rowlist if x != None)

            nbhoodExcX.append(sublist)

        #print("the L is ", nbhoodInhL)
    #print("the R is ", nbhoodInhR)
        #print("the diag list is", nbhoodExcX)
        # generate all y-axis excitation

        # Store these lists as global parameters as they can be used to quickly match the spiking collector neuron
        # with the corresponding pixel xy coordinates (same_disparity_indices)
        # TODO: think of a better way to encode pixels: closed form formula would be perfect
        # These are also used when connecting the spike sources to the network! (retina_proj_l, retina_proj_r)

        global _retina_proj_l, _retina_proj_r, same_disparity_indices

        _retina_proj_l = nbhoodInhL
        _retina_proj_r = nbhoodInhR
        same_disparity_indices = nbhoodExcX

        if verbose:
            print("INFO: Connecting neurons for internal excitation and inhibition.")
        connectInhL = []

        for row in _retina_proj_l:
            for pop in row:
                for nb in row:
                    if nb != pop:
                        connectInhL.append(
                            (pop, nb, self.cell_params['synaptic']['wCCi'], self.cell_params['synaptic']['dCCi']))

        for i in range(0, len(network)):
            ps.Projection(network[i][2], network[i][2], ps.FromListConnector(
                connectInhL), receptor_type='inhibitory')

        connectInhR = []

        for col in _retina_proj_r:
            for pop in col:
                for nb in col:
                    if nb != pop:
                        connectInhR.append(
                            (pop, nb, self.cell_params['synaptic']['wCCi'], self.cell_params['synaptic']['dCCi']))

        for i in range(0, len(network)):
            ps.Projection(network[i][2], network[i][2], ps.FromListConnector(
                connectInhR), receptor_type='inhibitory')

        connectExcX = []

        for diag in same_disparity_indices:
            for i in range(0, len(diag)):
                for j in range(1, self.cell_params['topological']['radius_e'] + 1):

                    if i+j < len(diag):
                        connectExcX.append(
                            (diag[i], diag[i+j], self.cell_params['synaptic']['wCCe'], self.cell_params['synaptic']['dCCe']))
                    if i-j >= 0:
                        connectExcX.append(
                            (diag[i], diag[i-j], self.cell_params['synaptic']['wCCe'], self.cell_params['synaptic']['dCCe']))

        for i in range(0, len(network)):
            ps.Projection(network[i][2], network[i][2], ps.FromListConnector(
                connectExcX), receptor_type='excitatory')

        print("connectExcX is ", connectExcX)
        connectExcY = []

        for i in range(0, len(network)):
            for j in range(1, self.cell_params['topological']['radius_e'] + 1):

                if i+j < len(network):
                    connectExcY.append([i, i+j])
                if i-j >= 0:
                    connectExcY.append([i, i-j])

        for l in connectExcY:
            i = l[0]
            j = l[1]

            ps.Projection(network[i][2], network[j][2], ps.OneToOneConnector(), ps.StaticSynapse(weight=self.cell_params['synaptic']['wCCe'],
                                                                                                 delay=self.cell_params['synaptic']['dCCe']),
                          receptor_type='excitatory')

        print("the connection excitation y  is", connectExcY)

    def _connect_spike_sources(self, retinae=None, verbose=False):

        if verbose:
            print("INFO: Connecting Spike Sources to Network.")

        global _retina_proj_l, _retina_proj_r

        # left is 0--dimensionRetinaY-1; right is dimensionRetinaY--dimensionRetinaY*2-1
        connListRetLBlockerL = []
        connListRetLBlockerR = []
        connListRetRBlockerL = []
        connListRetRBlockerR = []
        connListRetLCollector = []
        connListRetRCollector = []

        pixel = np.arange(
            self.dim_x*self.dim_y).reshape(self.dim_x, self.dim_y)
        raw_pixel = []
        for y in range(self.dim_y):
            list_raw = pixel[:, y].tolist()
            raw_pixel.append(list_raw)

        for raw in raw_pixel:
            LL = []
            LR = []
            LC = []
            for index, pixel in enumerate(raw):
                for bc in _retina_proj_l[index]:
                    LL.append((pixel, bc,
                               self.cell_params['synaptic']['wSaB'],
                               self.cell_params['synaptic']['dSaB']))
                    LR.append((pixel, bc,
                               self.cell_params['synaptic']['wSzB'],
                               self.cell_params['synaptic']['dSzB']))
                    LC.append((pixel, bc,
                               self.cell_params['synaptic']['wSC'],
                               self.cell_params['synaptic']['dSC']))

            connListRetLBlockerL.append(LL)
            connListRetLBlockerR.append(LR)
            connListRetLCollector.append(LC)

        for raw in raw_pixel:
            RR = []
            RL = []
            RC = []
            for index, pixel in enumerate(raw):
                for bc in _retina_proj_r[index]:
                    RR.append((pixel, bc,
                               self.cell_params['synaptic']['wSaB'],
                               self.cell_params['synaptic']['dSaB']))
                    RL.append((pixel, bc,
                               self.cell_params['synaptic']['wSzB'],
                               self.cell_params['synaptic']['dSzB']))
                    RC.append((pixel, bc,
                               self.cell_params['synaptic']['wSC'],
                               self.cell_params['synaptic']['dSC']))

            connListRetRBlockerR.append(RR)
            connListRetRBlockerL.append(RL)
            connListRetRCollector.append(RC)

        retinaLeft = retinae['left'].retina_pop
        retinaRight = retinae['right'].retina_pop

        for i in range(self.dim_y):

            ps.Projection(retinaLeft,
                          self.network[i][0],
                          ps.FromListConnector(connListRetLBlockerL[i]),
                          receptor_type='excitatory')

            ps.Projection(retinaLeft,
                          self.network[i][1],
                          ps.FromListConnector(connListRetLBlockerR[i]),
                          receptor_type='inhibitory')

            ps.Projection(retinaLeft,
                          self.network[i][2],
                          ps.FromListConnector(connListRetLCollector[i]),
                          receptor_type='excitatory')
            ps.Projection(retinaRight,
                          self.network[i][0],
                          ps.FromListConnector(connListRetRBlockerL[i]),
                          receptor_type='inhibitory')

            ps.Projection(retinaRight,
                          self.network[i][1],
                          ps.FromListConnector(connListRetRBlockerR[i]),
                          receptor_type='excitatory')

            ps.Projection(retinaRight,
                          self.network[i][2],
                          ps.FromListConnector(connListRetRCollector[i]),
                          receptor_type='excitatory')

        print("the coonection list of RetinaL with collector",
              connListRetLCollector)
        print("the connection list of retinaR with collector",
              connListRetRCollector)
        print("the retina_proj_l", _retina_proj_l)
        print("the retina_proj_r", _retina_proj_r)
        print("the raw pixel is", raw_pixel)

        # configure for the live input streaming if desired
        if not(retinae['left'].use_prerecorded_input and retinae['right'].use_prerecorded_input):
            from spynnaker_external_devices_plugin.pyNN.connections.spynnaker_live_spikes_connection import \
                SpynnakerLiveSpikesConnection

            all_retina_labels = retinaLeft.labels + retinaRight.labels
            self.live_connection_sender = SpynnakerLiveSpikesConnection(receive_labels=None, local_port=19999,
                                                                        send_labels=all_retina_labels)

            # this callback will be executed right after simulation.run() has been called. If simply a while True
            # is put there, the main thread will stuck there and will not complete the simulation.
            # One solution might be to start a thread/process which runs a "while is_running:" loop unless the main thread
            # sets the "is_running" to False.
            self.live_connection_sender.add_start_callback(
                all_retina_labels[0], self.start_injecting)

            import DVSReader as dvs
            # the port numbers might well be wrong
            self.dvs_stream_left = dvs.DVSReader(port=0,
                                                 label=retinaLeft.label,
                                                 live_connection=self.live_connection_sender)
            self.dvs_stream_right = dvs.DVSReader(port=1,
                                                  label=retinaRight.label,
                                                  live_connection=self.live_connection_sender)

            # start the threads, i.e. start reading from the DVS. However, nothing will be sent to the SNN.
            # See start_injecting
            self.dvs_stream_left.start()
            self.dvs_stream_right.start()

    def start_injecting(self):
        # start injecting into the SNN
        self.dvs_stream_left.start_injecting = True
        self.dvs_stream_right.start_injecting = True

    def get_network_dimensions(self):
        parameters = {'size': self.size,
                      'dim_x': self.dim_x,
                      'dim_y': self.dim_y,
                      'min_d': self.min_disparity,
                      'max_d': self.max_disparity}
        return parameters

    """ this method returns (and saves) a full list of spike times
    with the corresponding pixel location and disparities."""

    def descri(self):
        for i in self.network:
            i[2].describe(template='population_default.txt', engine='default')

    def get_spikes(self, sort_by_time=True, save_spikes=True):
        global same_disparity_indices, _retina_proj_l

        neo_per_population = [x[2].get_data(
            variables=["spikes"]) for x in self.network]
        spikes_per_population = [
            x.segments[0].spiketrains for x in neo_per_population]

        spikes = list()

        for index, spikes_pop in enumerate(spikes_per_population):
            y_coord = index
            for index_neuron, spiketrains in enumerate(spikes_pop):
                for l in range(len(_retina_proj_l)):
                    if index_neuron in _retina_proj_l[l]:
                        x_coord = l
                        break
                for d in range(len(same_disparity_indices)):
                    if index_neuron in same_disparity_indices[d]:
                        disp = d
                        break
                for spike in spiketrains:
                    spikes.append(
                        (round(spike, 1), x_coord+1, y_coord+1, disp))

        if sort_by_time:
            spikes.sort(key=lambda x: x[0])
        if save_spikes:
            if not os.path.exists("./spikes"):
                os.makedirs("./spikes")
            i = 0
            while os.path.exists("./spikes/{0}_{1}_spikes.dat".format(self.experiment_name, i)):
                i += 1
            with open('./spikes/{0}_{1}_spikes.dat'.format(self.experiment_name, i), 'w') as fs:
                # for open the last spike file
                self.i = i
                print('the value of i is', i)
                self._write_preamble(fs)
                fs.write("### DATA FORMAT ###\n"
                         "# Description: All spikes from the Collector Neurons are recorded. The disparity is inferred "
                         "from the Neuron ID. The disparity is calculated with the left camera as reference."
                         "The timestamp is dependent on the simulation parameters (simulation timestep).\n"
                         "# Each row contains: "
                         "Time -- x-coordinate -- y-coordinate -- disparity\n"
                         "### DATA START ###\n")
                for s in spikes:
                    fs.write(str(s[0]) + " " + str(s[1]) + " " +
                             str(s[2]) + " " + str(s[3]) + "\n")
                fs.write("### DATA END ###")
                fs.close()
        return spikes

    def _write_preamble(self, opened_file_descriptor):
        if opened_file_descriptor is not None:
            f = opened_file_descriptor
            f.write("### PREAMBLE START ###\n")
            f.write("# Experiment name: \n\t{0}\n".format(
                self.experiment_name))
            f.write("# Network parameters "
                    "(size in ensembles, x-dimension, y-dimension, minimum disparity, maximum disparity, "
                    "radius of excitation, radius of inhibition): "
                    "\n\t{0} {1} {2} {3} {4} {5} {6}\n".format(self.size, self.dim_x, self.dim_y,
                                                               self.min_disparity, self.max_disparity,
                                                               self.cell_params['topological']['radius_e'],
                                                               self.cell_params['topological']['radius_i']))
            f.write("# Neural parameters "
                    "(tau_excitation, tau_inhibition, tau_membrane, v_reset_blocker, v_reset_collector): "
                    "\n\t{0} {1} {2} {3} {4}\n".format(self.cell_params['neural']['tau_E'],
                                                       self.cell_params['neural']['tau_I'],
                                                       self.cell_params['neural']['tau_mem'],
                                                       self.cell_params['neural']['v_reset_blocker'],
                                                       self.cell_params['neural']['v_reset_collector']))
            f.write('# Synaptic parameters '
                    '(wBC, dBC, wSC, dSC, wSaB, dSaB, wSzB, dSzB, wCCi, dCCi, wCCe, dCCe): '
                    '\n\t{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'
                    .format(self.cell_params['synaptic']['wBC'],
                            self.cell_params['synaptic']['dBC'],
                            self.cell_params['synaptic']['wSC'],
                            self.cell_params['synaptic']['dSC'],
                            self.cell_params['synaptic']['wSaB'],
                            self.cell_params['synaptic']['dSaB'],
                            self.cell_params['synaptic']['wSzB'],
                            self.cell_params['synaptic']['dSzB'],
                            self.cell_params['synaptic']['wCCi'],
                            self.cell_params['synaptic']['dCCi'],
                            self.cell_params['synaptic']['wCCe'],
                            self.cell_params['synaptic']['dCCe']))
            f.write('# Comments: Caution: The synaptic parameters may vary according with '
                    'different simulation time steps. To understand the abbreviations for the '
                    'synaptic parameters, see the code documentation.\n')
            f.write("### PREAMBLE END ###\n")

    """ this method returns the accumulated spikes for each disparity as a list. It is not very useful except when
    the disparity sorting and formatting in the more general one get_spikes is not needed."""


'''
    def get_spikes_right(self, sort_by_time=True, save_spikes=True):
        global same_disparity_indices, _retina_proj_r
	neo_per_population = [x[1].get_data(variables=["spikes"]) for x in self.network]
        spikes_per_population = [x.segments[0].spiketrains for x in neo_per_population]

        #for test the format of the spikes
        print(len(spikes_per_population),'han')
        print(len(spikes_per_population[0]),'first')
        print(len(spikes_per_population[1]),'2')
        print(len(spikes_per_population[20]),'derniere')
        #

        spikes = list()
        # for each column population in the network, find the x,y coordinates corresponding to the neuron
        # and the disparity. Then write them in the list and sort it by the timestamp value.
        for col_index, col in enumerate(spikes_per_population, 0):  # it is 0-indexed
            # find the disparity
	    #for test the spike of the network
	    #print('the spiketrains of the collector',col)

            disp = self.min_disparity
            for d in range(0, self.max_disparity + 1):
                if col_index in same_disparity_indices[d]:
                    disp = d + self.min_disparity
                    break
	    x_coord = 0
            for p in range(0, self.dim_x):
                if col_index in _retina_proj_r[p]:
                    x_coord = p
                    break
            # for each spike in the population extract the timestamp and x,y coordinates
	    i=0
            for spiketrains in col:
                y_coord = i
		i=i+1
		for spike in spiketrains:
			#y_coord=spiketrains.index(spike) # and then should change with how to choice the y
		#i think here is not right and i should change it
                #y_coord = int(spike[0])
                	spikes.append((round(spike, 1), x_coord+1, y_coord+1, disp))	# pixel coordinates are 1-indexed
        if sort_by_time:
            spikes.sort(key=lambda x: x[0])
        if save_spikes:
            if not os.path.exists("./spikes"):
                os.makedirs("./spikes")
            i = 0
            while os.path.exists("./spikes/{0}_{1}_spikes_right.dat".format(self.experiment_name, i)):
                i += 1
            with open('./spikes/{0}_{1}_spikes_right.dat'.format(self.experiment_name, i), 'w') as fs:
		# for open the last spike file
		self.i=i
		print('the value of i is',i)
                self._write_preamble(fs)
                fs.write("### DATA FORMAT ###\n"
                        "# Description: All spikes from the Collector Neurons are recorded. The disparity is inferred "
                        "from the Neuron ID. The disparity is calculated with the left camera as reference."
                        "The timestamp is dependent on the simulation parameters (simulation timestep).\n"
                        "# Each row contains: "
                        "Time -- x-coordinate -- y-coordinate -- disparity\n"
                        "### DATA START ###\n")
                for s in spikes:
                    fs.write(str(s[0]) + " " + str(s[1]) + " " + str(s[2]) + " " + str(s[3]) + "\n")
                fs.write("### DATA END ###")
                fs.close()
        return spikes
'''
