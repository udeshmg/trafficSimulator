import numpy as np
from Road_Elements.Lane import Lane
from Road_Elements.Vehicles import VehicleBlock
from Stat_Reporter.StatReporter import Reporter

DEBUG = 3

class Road:

    def __init__(self, rid, num_of_lanes, timeToTravel=0):  # road id
        self.straight_v_list = np.empty(shape=0, dtype=VehicleBlock)  # vehicles turning left and straight
        self.right_Turn_v_list = np.empty(shape=0, dtype=VehicleBlock)  # vehicles turning right

        self.num_of_lanes = num_of_lanes
        self.is_direc_changing = False
        self.is_in_upstream_change = False
        self.is_out_upstream_change = False
        self.direc_change_counter = 0
        self.straight_road = None
        self.right_road = None
        self.outgoing_traffic = 0 # modify here

        self.straight_v_list_downstream = np.empty(shape=0, dtype=VehicleBlock)  # vehicles turning left and straight
        self.right_Turn_v_list_downstream = np.empty(shape=0, dtype=VehicleBlock)  # vehicles turning right
        self.vehiclesFromUpstream = []
        self.vehiclesFromDownstream = []
        self.timeToTravel = timeToTravel

        self.trafficImbalance = 0
        self.trafficImbalance_counter = 0

        self.checkConsistent = 12
        self.lane_capacity = 4
        self.enable_check = True

        self.upstream_id = 0
        self.downstream_id = 0

        self.roadConf = np.empty(shape=0,dtype=int)

        self.reporter = Reporter.getInstance()


        # Init Lanes
        # make first half of the lanes towards the intersection 'IN'
        # and second half of the lanes away from the intersection 'OUT'

        self.lanes = np.empty(shape=0, dtype=Lane)
        for i in range(num_of_lanes):
            direc = 'IN'
            if i >= num_of_lanes / 2:
                direc = 'OUT'
            # temp_lane = Lane(LANE_CAPACITY, i, direc, 'IDLE')
            self.lanes = np.append(self.lanes, Lane(self.lane_capacity, i, direc, 'IDLE'))
        self.id = rid

    def assign_intersection(self, direc, id):
        if direc == 'UPSTREAM':
            self.upstream_id = id
        else:
            self.downstream_id = id

    def get_in_lanes_num(self, id):
        count = 0
        for i in range(self.num_of_lanes):
            if (self.lanes[i].get_direction() == 'IN'):
                count += 1

        if id == self.downstream_id:
            count = self.num_of_lanes - count

        return count

    def getChangeDirectionFromRoad(self, originalAction, id):
        action = originalAction

        if id == self.downstream_id:
            if originalAction == 'OUT':
                action = 'IN'
            elif originalAction == 'IN':
                action = 'OUT'

        return action

    def change_direction(self, originalAction, id, depth=0):

        if not self.is_direc_changing:
            action = originalAction

            if id == self.downstream_id:
                if originalAction == 'OUT':
                    action = 'IN'
                elif originalAction == 'IN':
                    action = 'OUT'


            for i in range(self.num_of_lanes):
                if self.lanes[i].get_direction() == 'OUT':
                    if (action == 'IN'):
                        if i < 4:
                            self.lanes[i].change_direction('IN')
                            self.direc_change_counter = self.checkConsistent
                            if depth != 0:
                                print("Increased by two: ", self.id)
                                self.lanes[i+1].change_direction('IN')
                                self.direc_change_counter = int(1.5 * self.checkConsistent)
                            self.is_in_upstream_change = True
                            self.is_direc_changing = True

                            return True
                        else:
                            if DEBUG > 2:
                                print("Cannot Increase the input lanes")
                            return False
                    else:
                        if i > 2:
                            self.lanes[i - 1].change_direction('OUT')
                            self.direc_change_counter = self.checkConsistent
                            if depth != 0:
                                print("Increased by two: ", self.id)
                                self.lanes[i-2].change_direction('OUT')
                                self.direc_change_counter = int(1.5 * self.checkConsistent)
                            self.is_out_upstream_change = False
                            self.is_direc_changing = True

                            return True
                        else:
                            if DEBUG > 2:
                                print("Cannot Increase the Output lanes")
                            return False
        return False
    '''
        Add block: Add vehicle block to the road. 
        args: vb
    '''

    def add_block(self, vb, direc):

        #print("Vehicle added : RID", self.id, " Vehicle ID ", vb.id)
        if direc == 'DOWN':
            vb.setCurrentRoadDetails(self.id, 'DOWN')
            if True :#self.get_num_vehicles(self.downstream_id, vb.get_direction()) + vb.get_num_vehicles() <= 120:
                if vb.get_direction() == 'S' or vb.get_direction() == 'L':
                    self.straight_v_list_downstream = np.append(self.straight_v_list_downstream, vb)
                else:
                    self.right_Turn_v_list_downstream = np.append(self.right_Turn_v_list_downstream, vb)

                if DEBUG > 4:
                    print(" Number of vehicles added : ", self.id, vb.get_num_vehicles(), vb.get_direction())


        else:
            vb.setCurrentRoadDetails(self.id, 'UP')
            if True: # self.get_num_vehicles(self.upstream_id, vb.get_direction()) + vb.get_num_vehicles() <= 120:
                if vb.get_direction() == 'S' or vb.get_direction() == 'L':
                    self.straight_v_list = np.append(self.straight_v_list, vb)
                else:
                    self.right_Turn_v_list = np.append(self.right_Turn_v_list, vb)

                if DEBUG > 4:
                    print(" Number of vehicles added : ", self.id, vb.get_num_vehicles(), vb.get_direction())

        # else:
        #    self.straight_v_list = np.append(self.straight_v_list, VehicleBlock(40 - self.get_num_vehicles()))
        #    print(" Number of vehicles added (full) : ", self.id, 40 - self.get_num_vehicles())

    '''
        remove_block releases the vehicles from the vehicle blocks on specified direction
        until the capacity per unit time 
        args:
            action (signal light)
        returns:
            total waiting time and the number of outgoing vehicles

    '''

    def remove_block(self, vehiclesToRemove, action, id, is_first=True, straight_blocked=False, left_blocked=False):
        assert (action == 'R' or action == 'S')
        total_wait = 0
        total_vehi = 0
        remain_cap = vehiclesToRemove
        vehicles = np.empty(shape=0, dtype=VehicleBlock)
        vehicles_to_left = np.empty(shape=0, dtype=VehicleBlock)



        if self.is_direc_changing:

            if self.is_in_upstream_change:
                if id == self.upstream_id:
                    remain_cap = self.capacity(self.upstream_id, 'IN') - self.lane_capacity
                    print(" Upstream:", self.id, " Input increasing: ", remain_cap)
                else:
                    remain_cap = self.capacity(self.downstream_id, 'IN') * 0.5
                    print(" Downstream:", self.id, "Output decreasing: ", remain_cap)

            elif self.is_out_upstream_change:
                if id == self.downstream_id:
                    remain_cap = self.capacity(self.downstream_id, 'IN') - self.lane_capacity
                    print(" Downstream:", self.id, " Input increasing: ", remain_cap)
                else:
                    remain_cap = self.capacity(self.upstream_id, 'IN') * 0.5
                    print(" Upstream:", self.id, " Output increasing: ", remain_cap)
            remain_cap = int(remain_cap)

        if not is_first:
            remain_cap = int(remain_cap * 1.5)

        if DEBUG > 4:
            print('Capacity: ', self.capacity('IN'))

        pointer = 0

        if id == self.downstream_id:
            if (action == 'S'):  # remove vehicles in straight direction
                if self.straight_v_list_downstream.size == 0:
                    return 0, 0, np.empty(shape=0, dtype=VehicleBlock), np.empty(shape=0, dtype=VehicleBlock)
                while remain_cap > 0 and pointer < self.straight_v_list_downstream.size:  # if first vehicle block has less vehicles than current capacity
                    # then loop until remain capacity becomes zero.
                    if self.straight_v_list_downstream[pointer].get_num_vehicles() <= remain_cap:
                        temp_a, temp_b = self.straight_v_list_downstream[pointer].get_current_time() * self.straight_v_list_downstream[
                            pointer].get_num_vehicles(), \
                                         self.straight_v_list_downstream[pointer].get_num_vehicles()

                        if self.straight_v_list_downstream[pointer].turn_direction.size != 0:
                            if self.straight_v_list_downstream[pointer].get_direction() == 'S':
                                if not straight_blocked:
                                    self.straight_v_list_downstream[pointer].update_direction()
                                    self.straight_v_list_downstream[pointer].resetTime()
                                    vehicles = np.append(vehicles, self.straight_v_list_downstream[pointer])

                                    self.straight_v_list_downstream = np.delete(self.straight_v_list_downstream,
                                                                                pointer)  # delete block in queue with index
                                    total_wait += temp_a
                                    total_vehi += temp_b
                                    remain_cap -= temp_b
                                else:
                                    pointer += 1
                            else:
                                if not left_blocked:
                                    self.straight_v_list_downstream[pointer].update_direction()
                                    self.straight_v_list_downstream[pointer].resetTime()
                                    vehicles_to_left = np.append(vehicles_to_left, self.straight_v_list_downstream[pointer])

                                    self.straight_v_list_downstream = np.delete(self.straight_v_list_downstream,
                                                                                pointer)  # delete block in queue with index
                                    total_wait += temp_a
                                    total_vehi += temp_b
                                    remain_cap -= temp_b
                                else:
                                    pointer += 1





                    else:
                        #self.straight_v_list_downstream[0].reduce_vehicles(remain_cap)

                        if self.straight_v_list_downstream[pointer].turn_direction.size != 0:
                            if self.straight_v_list_downstream[pointer].get_direction() == 'S':
                                if not straight_blocked:
                                    self.straight_v_list_downstream[pointer].reduce_vehicles(remain_cap)
                                    vb = VehicleBlock(remain_cap, self.straight_v_list_downstream[pointer].get_route(),
                                                      self.straight_v_list_downstream[pointer].id)
                                    vb.update_direction()
                                    vb.abs_time = self.straight_v_list_downstream[pointer].abs_time
                                    vehicles = np.append(vehicles, vb)

                                    total_wait += self.straight_v_list_downstream[pointer].get_current_time() * remain_cap
                                    total_vehi += remain_cap
                                    remain_cap = 0
                                else:
                                    pointer += 1
                            else:
                                if not left_blocked:
                                    self.straight_v_list_downstream[pointer].reduce_vehicles(remain_cap)
                                    vb = VehicleBlock(remain_cap, self.straight_v_list_downstream[pointer].get_route())
                                    vb.update_direction()
                                    vb.abs_time = self.straight_v_list_downstream[pointer].abs_time
                                    vehicles_to_left = np.append(vehicles_to_left, vb)

                                    total_wait += self.straight_v_list_downstream[pointer].get_current_time() * remain_cap
                                    total_vehi += remain_cap
                                    remain_cap = 0
                                else:
                                    pointer += 1





            else:
                if self.right_Turn_v_list_downstream.size == 0:
                    return 0, 0, np.empty(shape=0, dtype=VehicleBlock), np.empty(shape=0, dtype=VehicleBlock)
                while remain_cap > 0 and self.right_Turn_v_list_downstream.size:
                    if self.right_Turn_v_list_downstream[0].get_num_vehicles() <= remain_cap:
                        temp_a, temp_b = self.right_Turn_v_list_downstream[0].get_current_time() * self.right_Turn_v_list_downstream[
                            0].get_num_vehicles(), \
                                         self.right_Turn_v_list_downstream[0].get_num_vehicles()

                        self.right_Turn_v_list_downstream[0].update_direction()
                        self.right_Turn_v_list_downstream[0].resetTime()
                        vehicles = np.append(vehicles, self.right_Turn_v_list_downstream[0])

                        self.right_Turn_v_list_downstream = np.delete(self.right_Turn_v_list_downstream, 0)
                        total_wait += temp_a
                        total_vehi += temp_b
                        remain_cap -= temp_b
                    else:
                        self.right_Turn_v_list_downstream[0].reduce_vehicles(remain_cap)

                        vb = VehicleBlock(remain_cap, self.right_Turn_v_list_downstream[0].get_route())
                        vb.update_direction()
                        vb.abs_time = self.right_Turn_v_list_downstream[0].abs_time
                        vehicles = np.append(vehicles, vb)

                        total_wait += self.right_Turn_v_list_downstream[0].get_current_time() * remain_cap
                        total_vehi += remain_cap
                        remain_cap = 0

        else:
            if (action == 'S'):  # remove vehicles in straight direction
                if self.straight_v_list.size == 0:
                    return 0, 0, np.empty(shape=0, dtype=VehicleBlock), np.empty(shape=0, dtype=VehicleBlock)
                while remain_cap > 0 and pointer < self.straight_v_list.size:  # if first vehicle block has less vehicles than current capacity
                    # then loop until remain capacity becomes zero.
                    if self.straight_v_list[pointer].get_num_vehicles() <= remain_cap:
                        temp_a, temp_b = self.straight_v_list[pointer].get_current_time() * self.straight_v_list[
                            pointer].get_num_vehicles(), \
                                         self.straight_v_list[pointer].get_num_vehicles()

                        if self.straight_v_list[pointer].turn_direction.size != 0:
                            if self.straight_v_list[pointer].get_direction() == 'S':
                                if not straight_blocked:
                                    self.straight_v_list[pointer].update_direction()
                                    self.straight_v_list[pointer].resetTime()
                                    vehicles = np.append(vehicles, self.straight_v_list[pointer])

                                    self.straight_v_list = np.delete(self.straight_v_list,
                                                                                pointer)  # delete block in queue with index
                                    total_wait += temp_a
                                    total_vehi += temp_b
                                    remain_cap -= temp_b
                                else:
                                    pointer += 1
                            else:
                                if not left_blocked:
                                    self.straight_v_list[pointer].update_direction()
                                    self.straight_v_list[pointer].resetTime()
                                    vehicles_to_left = np.append(vehicles_to_left, self.straight_v_list[pointer])

                                    self.straight_v_list = np.delete(self.straight_v_list,
                                                                                pointer)  # delete block in queue with index
                                    total_wait += temp_a
                                    total_vehi += temp_b
                                    remain_cap -= temp_b
                                else:
                                    pointer += 1

                    else:

                        if self.straight_v_list[pointer].turn_direction.size != 0:
                            if self.straight_v_list[pointer].get_direction() == 'S':
                                if not straight_blocked:
                                    self.straight_v_list[pointer].reduce_vehicles(remain_cap)
                                    vb = VehicleBlock(remain_cap, self.straight_v_list[pointer].get_route())
                                    vb.update_direction()
                                    vb.abs_time = self.straight_v_list[pointer].abs_time
                                    vehicles = np.append(vehicles, vb)

                                    total_wait += self.straight_v_list[pointer].get_current_time() * remain_cap
                                    total_vehi += remain_cap
                                    remain_cap = 0
                                else:
                                    pointer += 1
                            else:
                                if not left_blocked:
                                    self.straight_v_list[pointer].reduce_vehicles(remain_cap)
                                    vb = VehicleBlock(remain_cap, self.straight_v_list[pointer].get_route())
                                    vb.update_direction()
                                    vb.abs_time = self.straight_v_list[pointer].abs_time
                                    vehicles_to_left = np.append(vehicles_to_left, vb)

                                    total_wait += self.straight_v_list[pointer].get_current_time() * remain_cap
                                    total_vehi += remain_cap
                                    remain_cap = 0
                                else:
                                    pointer += 1

            else:
                if self.right_Turn_v_list.size == 0:
                    return 0, 0, np.empty(shape=0, dtype=VehicleBlock), np.empty(shape=0, dtype=VehicleBlock)
                while remain_cap > 0 and self.right_Turn_v_list.size:
                    if self.right_Turn_v_list[0].get_num_vehicles() <= remain_cap:
                        temp_a, temp_b = self.right_Turn_v_list[0].get_current_time() * self.right_Turn_v_list[
                            0].get_num_vehicles(), \
                                         self.right_Turn_v_list[0].get_num_vehicles()

                        self.right_Turn_v_list[0].update_direction()
                        self.right_Turn_v_list[0].resetTime()
                        vehicles = np.append(vehicles, self.right_Turn_v_list[0])

                        self.right_Turn_v_list = np.delete(self.right_Turn_v_list, 0)
                        total_wait += temp_a
                        total_vehi += temp_b
                        remain_cap -= temp_b
                    else:
                        self.right_Turn_v_list[0].reduce_vehicles(remain_cap)

                        vb = VehicleBlock(remain_cap, self.right_Turn_v_list[0].get_route())
                        vb.update_direction()
                        vb.abs_time = self.right_Turn_v_list[0].abs_time
                        vehicles = np.append(vehicles, vb)

                        total_wait += self.right_Turn_v_list[0].get_current_time() * remain_cap
                        total_vehi += remain_cap
                        remain_cap = 0

        return total_wait, total_vehi, vehicles, vehicles_to_left

    def get_wait_time(self, direc, id):
        time = 0

        if id == self.upstream_id:
            if direc == 'S':
                for i in range(0, self.straight_v_list.size):
                    time += self.straight_v_list[i].waiting_time()
                # info(2, ' Road id : ' + str(self.id) + ' Wait time: ' + str(time))
            elif direc == 'T':
                for i in range(0, self.straight_v_list.size):
                    time += self.straight_v_list[i].waiting_time()
                for i in range(0, self.right_Turn_v_list.size):
                    time += self.right_Turn_v_list[i].waiting_time()
            else: # direc  == 'R'
                for i in range(0, self.right_Turn_v_list.size):
                    time += self.right_Turn_v_list[i].waiting_time()

        else:
            if direc == 'S':
                for i in range(0, self.straight_v_list_downstream.size):
                    time += self.straight_v_list_downstream[i].waiting_time()

            elif direc == 'T':
                for i in range(0, self.straight_v_list_downstream.size):
                    time += self.straight_v_list_downstream[i].waiting_time()
                for i in range(0, self.right_Turn_v_list_downstream.size):
                    time += self.right_Turn_v_list_downstream[i].waiting_time()
            else:
                for i in range(0, self.right_Turn_v_list_downstream.size):
                    time += self.right_Turn_v_list_downstream[i].waiting_time()

        return time

    def get_num_vehicles(self, id, direc='T'):
        total = 0

        if id == self.downstream_id:

            if direc == 'S':
                for i in range(0, self.straight_v_list_downstream.size):
                    total += self.straight_v_list_downstream[i].get_num_vehicles()
            elif direc == 'R':
                for i in range(0, self.right_Turn_v_list_downstream.size):
                    total += self.right_Turn_v_list_downstream[i].get_num_vehicles()
            else:
                for i in range(0, self.right_Turn_v_list_downstream.size):
                    total += self.right_Turn_v_list_downstream[i].get_num_vehicles()
                for i in range(0, self.straight_v_list_downstream.size):
                    total += self.straight_v_list_downstream[i].get_num_vehicles()
        # info(2, 'Road id : ' + str(self.id) + ' Number of vehicles: ' + str(total))
            if DEBUG > 4:
                print('Road, vehicles ', direc, self.id, total)

        else:
            if direc == 'S':
                for i in range(0, self.straight_v_list.size):
                    total += self.straight_v_list[i].get_num_vehicles()
            elif direc == 'R':
                for i in range(0, self.right_Turn_v_list.size):
                    total += self.right_Turn_v_list[i].get_num_vehicles()
            else:
                for i in range(0, self.right_Turn_v_list.size):
                    total += self.right_Turn_v_list[i].get_num_vehicles()
                for i in range(0, self.straight_v_list.size):
                    total += self.straight_v_list[i].get_num_vehicles()
            # info(2, 'Road id : ' + str(self.id) + ' Number of vehicles: ' + str(total))
            if DEBUG > 4:
                print('Road, vehicles ', direc, self.id, total)

        return total

    def get_outgoing_vehicles(self, id, direc='T'):
        total = 0

        if id == self.upstream_id:

            if direc == 'S':
                for i in range(0, self.straight_v_list_downstream.size):
                    total += self.straight_v_list_downstream[i].get_num_vehicles()
            elif direc == 'R':
                for i in range(0, self.right_Turn_v_list_downstream.size):
                    total += self.right_Turn_v_list_downstream[i].get_num_vehicles()
            else:
                for i in range(0, self.right_Turn_v_list_downstream.size):
                    total += self.right_Turn_v_list_downstream[i].get_num_vehicles()
                for i in range(0, self.straight_v_list_downstream.size):
                    total += self.straight_v_list_downstream[i].get_num_vehicles()
        # info(2, 'Road id : ' + str(self.id) + ' Number of vehicles: ' + str(total))
            if DEBUG > 4:
                print('Road, vehicles ', direc, self.id, total)

        else:
            if direc == 'S':
                for i in range(0, self.straight_v_list.size):
                    total += self.straight_v_list[i].get_num_vehicles()
            elif direc == 'R':
                for i in range(0, self.right_Turn_v_list.size):
                    total += self.right_Turn_v_list[i].get_num_vehicles()
            else:
                for i in range(0, self.right_Turn_v_list.size):
                    total += self.right_Turn_v_list[i].get_num_vehicles()
                for i in range(0, self.straight_v_list.size):
                    total += self.straight_v_list[i].get_num_vehicles()
            # info(2, 'Road id : ' + str(self.id) + ' Number of vehicles: ' + str(total))
            if DEBUG > 4:
                print('Road, vehicles ', direc, self.id, total)

        return total

    '''
        Advances the clock of the road
    '''

    def step(self, time_step):

        if (self.direc_change_counter > 0):
            self.direc_change_counter -= 1
        else:
            self.is_direc_changing = False
            self.is_out_upstream_change = False
            self.is_in_upstream_change = False

        for i in range(0, self.straight_v_list.size):
            self.straight_v_list[i].step(time_step)

        for i in range(0, self.right_Turn_v_list.size):
            self.right_Turn_v_list[i].step(time_step)

        for i in range(0, self.straight_v_list_downstream.size):
            self.straight_v_list_downstream[i].step(time_step)

        for i in range(0, self.right_Turn_v_list_downstream.size):
            self.right_Turn_v_list_downstream[i].step(time_step)

        for i in range(len(self.vehiclesFromUpstream)):
            self.vehiclesFromUpstream[i][0].absStep(time_step)
            self.vehiclesFromUpstream[i][1] +=  time_step

        for i in range(len(self.vehiclesFromDownstream)):
            self.vehiclesFromDownstream[i][0].absStep(time_step)
            self.vehiclesFromDownstream[i][1] +=  time_step

        self.removeFromTravellingQueue()

        # self.remove_outgoing_vehicles(self.capacity('OUT'))

        if (self.get_num_vehicles(self.downstream_id) - self.get_num_vehicles(self.upstream_id)) / max(self.get_num_vehicles(self.downstream_id), self.get_num_vehicles(self.upstream_id),
                                                                   20) > 0.47:
            self.trafficImbalance_counter = min(60, self.trafficImbalance_counter + 1)  # out going traffic high
        if (self.get_num_vehicles(self.upstream_id)- self.get_num_vehicles(self.downstream_id)) / max(self.get_num_vehicles(self.downstream_id), self.get_num_vehicles(self.upstream_id),
                                                                   20) > 0.47:
            self.trafficImbalance_counter = max(-60, self.trafficImbalance_counter - 1)

        if (self.trafficImbalance_counter > 0) and (self.get_num_vehicles(self.upstream_id) - self.get_num_vehicles(self.downstream_id)) > 2:
            self.trafficImbalance_counter = 0
        if (self.trafficImbalance_counter < 0) and (self.get_num_vehicles(self.downstream_id) - self.get_num_vehicles(self.upstream_id)) > 2:
            self.trafficImbalance_counter = 0
        if self.get_num_vehicles(self.downstream_id) == self.get_num_vehicles(self.upstream_id):
            self.trafficImbalance_counter = 0

        # print("Imbalance counter", self.id, self.trafficImbalance_counter)

        self.get_traffic_details()
        self.roadConf = np.append(self.roadConf, (self.get_in_lanes_num(self.upstream_id)))

    '''
        Capacity of the Road in one direction 
        args:
            direc : Direction 'IN' or 'OUT'
            'IN' towards the junction and 'OUT' for outgoing from the junction
    '''

    def capacity(self, id, direc='IN'):
        assert (direc == 'IN' or direc == 'OUT')
        cap = 0
        for i in range(self.num_of_lanes):
            if DEBUG > 5:
                print("Lane direction", i, self.lanes[i].get_direction())

            if self.lanes[i].get_direction() == direc:
                cap += self.lanes[i].get_capacity()
        if DEBUG > 5:
            print("Lane and capacity ", direc, self.id, cap)

        if id == self.downstream_id:
            cap = self.num_of_lanes*self.lane_capacity - cap

        return cap

    def reset(self):

        while len(self.right_Turn_v_list) != 0:
            self.right_Turn_v_list = np.delete(self.right_Turn_v_list, 0)
        while len(self.straight_v_list) != 0:
            self.straight_v_list = np.delete(self.straight_v_list, 0)
        while len(self.right_Turn_v_list_downstream) != 0:
            self.right_Turn_v_list = np.delete(self.right_Turn_v_list_downstream, 0)
        while len(self.straight_v_list_downstream) != 0:
            self.straight_v_list = np.delete(self.straight_v_list_downstream, 0)

        for i in range(self.num_of_lanes):
            direc = 'IN'
            if i >= self.num_of_lanes / 2:
                direc = 'OUT'
            # temp_lane = Lane(LANE_CAPACITY, i, direc, 'IDLE')
            self.lanes[i].change_direction(direc)

        self.trafficImbalance = 0
        self.trafficImbalance_counter = 0

        print("Rest completed for road: ", self.id)

    def get_traffic_details(self):
        if DEBUG > 3:
            print("Incoming traffic of road", self.id, self.get_num_vehicles(self.upstream_id))
            print("Outgoing traffic of road", self.id, self.get_outgoing_vehicles(self.upstream_id))
        return self.outgoing_traffic

    def set_outgoing_traffic(self, num_vehicles):
        self.outgoing_traffic = min(260, self.outgoing_traffic + num_vehicles)

    def set_outgoing_vb(self,vb,id):
        if id == self.upstream_id:
            while vb.size != 0:
                temp = vb[0]
                vb = np.delete(vb,0)
                if temp.turn_direction.size != 0:
                    temp.setCurrentRoadDetails(self.id, 'DOWN')
                    self.vehiclesFromUpstream.append([temp, 0])
                else:
                    temp.finaliseRoute()
        else:
            while vb.size != 0:
                temp = vb[0]
                vb = np.delete(vb,0)
                if temp.turn_direction.size != 0:
                    temp.setCurrentRoadDetails(self.id, 'UP')
                    self.vehiclesFromDownstream.append([temp, 0])
                else:
                    temp.finaliseRoute()

        return 0,0


    def removeFromTravellingQueue(self):
        timeStepReached = False
        vbs = np.empty(shape=0, dtype=VehicleBlock)
        while (not timeStepReached) and len(self.vehiclesFromUpstream) != 0:
            if self.vehiclesFromUpstream[0][1] >= self.timeToTravel:
                vbs = np.append(vbs, self.vehiclesFromUpstream.pop(0)[0])
            else:
                timeStepReached = True



        self.moveTowaitingQueue(vbs, self.upstream_id)

        timeStepReached = False
        vbs = np.empty(shape=0, dtype=VehicleBlock)
        while (not timeStepReached) and len(self.vehiclesFromDownstream) != 0:
            if self.vehiclesFromDownstream[0][1] >= self.timeToTravel:
                vbs = np.append(vbs, self.vehiclesFromDownstream.pop(0)[0])
            else:
                timeStepReached = True


        self.moveTowaitingQueue(vbs, self.downstream_id)


    def moveTowaitingQueue(self, vb, id):
        time = 0
        vehi = 0
        if id == self.upstream_id:
            for i in range(vb.size):
                if vb[i].turn_direction.size != 0:
                    if (vb[i].get_direction() == 'S') or (vb[i].get_direction() == 'L'):
                        self.straight_v_list_downstream = np.append(self.straight_v_list_downstream,vb[i])
                    elif vb[i].get_direction() == 'R':
                        self.right_Turn_v_list_downstream = np.append(self.right_Turn_v_list_downstream,vb[i])
                    else:
                        vb[i].finaliseRoute()
                else:
                    vb[i].finaliseRoute()



        else:
            for i in range(vb.size):
                if vb[i].turn_direction.size != 0:
                    if vb[i].get_direction() == 'S' or vb[i].get_direction() == 'L':
                        self.straight_v_list = np.append(self.straight_v_list,vb[i])
                    else:
                        self.right_Turn_v_list = np.append(self.right_Turn_v_list,vb[i])
                else:
                    vb[i].finaliseRoute()

        return time, vehi

    def remove_outgoing_vehicles(self, num_vehicles):
        if DEBUG > 3:
            print("Remove vehicles", self.id, num_vehicles)
        if self.outgoing_traffic < num_vehicles:
            temp = self.outgoing_traffic
            self.outgoing_traffic = 0
            return temp
        else:
            self.outgoing_traffic -= num_vehicles
            return num_vehicles

    def is_road_idle(self):
        # return not self.is_direc_changing
        if self.trafficImbalance_counter == 50:
            return True
        else:
            return False

    def get_change_counter(self):
        return self.direc_change_counter

    def set_straight_road(self, road):
        # print("Straight Connection between", self.id, " and ", road.get_id())
        self.straight_road = road

    def set_right_road(self, road):
        # print("Right turn Connection between", self.id, " and ", road.get_id())
        self.right_road = road

    def get_traffic_imbalance(self, id):
        if self.trafficImbalance_counter >= 30:
            if id == self.upstream_id:
                return 1  # outgoing traffic high
            else:
                return 2
        elif self.trafficImbalance_counter <= -30:
            if id == self.upstream_id:
                return 2
            else:
                return 1
        else:
            return 0

    def setRoadConfigurationData(self):
        self.reporter.setRoadData(self.id, self.roadConf)

    def get_id(self):
        return self.id


