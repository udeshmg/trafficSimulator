from tkinter import *
from tkinter import ttk
import random
from collections import deque

# simulation
# create lanes

class SimulatorTraffic:

    def __init__(self,gui):
        self.canvas = Canvas(gui, width=600,height=600,bg='white')
        self.canvas.pack()
        self.rd_1_lane_1_start_x = 260
        self.rd_1_lane_1_start_y = 260
        self.rd_1_lane_1_end_x = self.rd_1_lane_1_start_x - 120
        self.rd_1_lane_1_end_y = self.rd_1_lane_1_start_y + 20
        self.rd_1_lane_1 = self.canvas.create_rectangle(self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y, self.rd_1_lane_1_start_x, self.rd_1_lane_1_start_y,
                                              fill='gray')

        self.rd_1_lane_2_start_x = 260
        self.rd_1_lane_2_start_y = 280
        self.rd_1_lane_2_end_x = self.rd_1_lane_2_start_x - 120
        self.rd_1_lane_2_end_y = self.rd_1_lane_2_start_y + 20
        self.rd_1_lane_2 = self.canvas.create_rectangle(self.rd_1_lane_2_end_x, self.rd_1_lane_2_end_y, self.rd_1_lane_2_start_x, self.rd_1_lane_2_start_y,
                                              fill='gray')

        #self.canvas.create_rectangle(self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y+60, self.rd_1_lane_1_start_x, self.rd_1_lane_1_start_y+40,
                                              #fill='#9E9CA5')

        self.rd_2_lane_1_start_x = 320
        self.rd_2_lane_1_start_y = 260
        self.rd_2_lane_1_end_x = self.rd_2_lane_1_start_x + 20
        self.rd_2_lane_1_end_y = self.rd_2_lane_1_start_y - 120
        self.rd_2_lane_1 = self.canvas.create_rectangle(self.rd_2_lane_1_end_x, self.rd_2_lane_1_end_y, self.rd_2_lane_1_start_x, self.rd_2_lane_1_start_y,
                                              fill='gray')

        self.rd_2_lane_2_start_x = 300
        self.rd_2_lane_2_start_y = 260
        self.rd_2_lane_2_end_x = self.rd_2_lane_2_start_x + 20
        self.rd_2_lane_2_end_y = self.rd_2_lane_2_start_y - 120
        self.rd_2_lane_2 = self.canvas.create_rectangle(self.rd_2_lane_2_end_x, self.rd_2_lane_2_end_y, self.rd_2_lane_2_start_x, self.rd_2_lane_2_start_y,
                                              fill='gray')

        #self.canvas.create_rectangle(self.rd_2_lane_1_end_x-60, self.rd_2_lane_1_end_y, self.rd_2_lane_1_start_x-40, self.rd_2_lane_1_start_y,
                                              #fill='gray')

        self.rd_3_lane_1_start_x = 340
        self.rd_3_lane_1_start_y = 320
        self.rd_3_lane_1_end_x = self.rd_3_lane_1_start_x + 120
        self.rd_3_lane_1_end_y = self.rd_3_lane_1_start_y + 20
        self.rd_3_lane_1 = self.canvas.create_rectangle(self.rd_3_lane_1_end_x, self.rd_3_lane_1_end_y, self.rd_3_lane_1_start_x, self.rd_3_lane_1_start_y,
                                              fill='gray')

        self.rd_3_lane_2_start_x = 340
        self.rd_3_lane_2_start_y = 300
        self.rd_3_lane_2_end_x = self.rd_3_lane_2_start_x + 120
        self.rd_3_lane_2_end_y = self.rd_3_lane_2_start_y + 20
        self.rd_3_lane_2 = self.canvas.create_rectangle(self.rd_3_lane_2_end_x, self.rd_3_lane_2_end_y, self.rd_3_lane_2_start_x, self.rd_3_lane_2_start_y,
                                              fill='gray')

        self.rd_4_lane_1_start_x = 260
        self.rd_4_lane_1_start_y = 340
        self.rd_4_lane_1_end_x = self.rd_4_lane_1_start_x + 20
        self.rd_4_lane_1_end_y = self.rd_4_lane_1_start_y + 120
        self.rd_4_lane_1 = self.canvas.create_rectangle(self.rd_4_lane_1_end_x, self.rd_4_lane_1_end_y, self.rd_4_lane_1_start_x, self.rd_4_lane_1_start_y,
                                              fill='gray')

        self.rd_4_lane_2_start_x = 280
        self.rd_4_lane_2_start_y = 340
        self.rd_4_lane_2_end_x = self.rd_4_lane_2_start_x + 20
        self.rd_4_lane_2_end_y = self.rd_4_lane_2_start_y + 120
        self.rd_4_lane_2 = self.canvas.create_rectangle(self.rd_4_lane_2_end_x, self.rd_4_lane_2_end_y, self.rd_4_lane_2_start_x, self.rd_4_lane_2_start_y,
                                              fill='gray')

        cross = self.canvas.create_rectangle(260, 260, 340, 340, fill='yellow')


        self.cross1 = self.canvas.create_rectangle(265, 265, 270, 270, fill='red')
        self.cross2 = self.canvas.create_rectangle(265, 290, 270, 295, fill='red')
        self.cross3 = self.canvas.create_rectangle(303, 265, 308, 270, fill='red')
        self.cross4 = self.canvas.create_rectangle(325, 265, 330, 270, fill='red')
        self.cross5 = self.canvas.create_rectangle(325, 305, 330, 310, fill='red')
        self.cross6 = self.canvas.create_rectangle(325, 330, 330, 335, fill='red')
        self.cross7 = self.canvas.create_rectangle(265, 330, 270, 335, fill='red')
        self.cross8 = self.canvas.create_rectangle(291, 330, 296, 335, fill='red')



        self.vehicle_blocks_W = deque(maxlen=100)
        self.vehicle_blocks_E = deque(maxlen=100)
        self.vehicle_blocks_S = deque(maxlen=100)
        self.vehicle_blocks_N = deque(maxlen=100)

        self.vehicle_blocks_W_R = deque(maxlen=100)
        self.vehicle_blocks_E_R = deque(maxlen=100)
        self.vehicle_blocks_S_R = deque(maxlen=100)
        self.vehicle_blocks_N_R = deque(maxlen=100)

        self.iter_box = self.canvas.create_text(10, 50, anchor=W, font=('Arial', 20),
                                                 text='Start')

        self.wait = self.canvas.create_text(100, 550, anchor=W, font=('Arial', 20),
                                            text='Wait time')

        self.through = self.canvas.create_text(100, 600, anchor=W, font=('Arial', 20),
                                               text='throughput')

        self.sig_pattern = self.canvas.create_text(100, 650, anchor=W, font=('Arial', 20),
                                                   text='Phase')

        self.lane_conf1 =  self.canvas.create_text(280, 110, anchor=W, font=('Arial', 15),
                                                    text='1')
        self.lane_conf2 = self.canvas.create_text(30, 270, anchor=W, font=('Arial', 15),
                                text='2')

        self.lane_conf3 = self.canvas.create_text(240, 500, anchor=W, font=('Arial', 15),
                            text='3')

        self.lane_conf4 = self.canvas.create_text(490, 310, anchor=W, font=('Arial', 15),
                                text='4')


    def simulator(self):
        self.rd_1_lane_1_end_x = self.rd_1_lane_1_start_x - 120
        self.rd_1_lane_2_end_x = self.rd_1_lane_2_start_x - 120
        self.rd_2_lane_1_end_y = self.rd_2_lane_1_start_y - 120
        self.rd_2_lane_2_end_y = self.rd_2_lane_2_start_y - 120
        self.rd_3_lane_1_end_x = self.rd_3_lane_1_start_x + 120
        self.rd_3_lane_2_end_x = self.rd_3_lane_2_start_x + 120
        self.rd_4_lane_1_end_y = self.rd_4_lane_1_start_y + 120
        self.rd_4_lane_2_end_y = self.rd_4_lane_2_start_y + 120

        self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y, self.rd_1_lane_1_start_x,
                      self.rd_1_lane_1_start_y)
        self.canvas.coords(self.rd_1_lane_2, self.rd_1_lane_2_end_x, self.rd_1_lane_2_end_y, self.rd_1_lane_2_start_x,
                      self.rd_1_lane_2_start_y)
        self.canvas.coords(self.rd_2_lane_1, self.rd_2_lane_1_end_x, self.rd_2_lane_1_end_y, self.rd_2_lane_1_start_x,
                      self.rd_2_lane_1_start_y)
        self.canvas.coords(self.rd_2_lane_2, self.rd_2_lane_2_end_x, self.rd_2_lane_2_end_y, self.rd_2_lane_2_start_x,
                      self.rd_2_lane_2_start_y)
        self.canvas.coords(self.rd_3_lane_1, self.rd_3_lane_1_end_x, self.rd_3_lane_1_end_y, self.rd_3_lane_1_start_x,
                      self.rd_3_lane_1_start_y)
        self.canvas.coords(self.rd_3_lane_2, self.rd_3_lane_2_end_x, self.rd_3_lane_2_end_y, self.rd_3_lane_2_start_x,
                      self.rd_3_lane_2_start_y)
        self.canvas.coords(self.rd_4_lane_1, self.rd_4_lane_1_end_x, self.rd_4_lane_1_end_y, self.rd_4_lane_1_start_x,
                      self.rd_4_lane_1_start_y)
        self.canvas.coords(self.rd_4_lane_2, self.rd_4_lane_2_end_x, self.rd_4_lane_2_end_y, self.rd_4_lane_2_start_x,
                      self.rd_4_lane_2_start_y)
        self.canvas.after(20)
        self.canvas.update()

    def lane_conf(self,conf):
        self.canvas.delete(self.lane_conf1)
        self.lane_conf1 =  self.canvas.create_text(300, 120, anchor=W, font=('Arial', 15),
                                                    text=str(conf[3]))
        self.canvas.delete(self.lane_conf2)
        self.lane_conf2 = self.canvas.create_text(122, 270, anchor=W, font=('Arial', 15),
                                text=str(conf[0]))
        self.canvas.delete(self.lane_conf3)
        self.lane_conf3 = self.canvas.create_text(260, 488, anchor=W, font=('Arial', 15),
                            text=str(conf[1]))
        self.canvas.delete(self.lane_conf4)
        self.lane_conf4 = self.canvas.create_text(480, 315, anchor=W, font=('Arial', 15),
                                text=str(conf[2]))

    def update_iter(self,iter, wait_time, throughput, action):
        self.canvas.delete(self.iter_box)
        string = 'Iteration: ' + str(iter)
        self.iter_box = self.canvas.create_text(10, 50, anchor=W, font=('Arial', 20),
                                                 text=string)

        self.canvas.delete(self.wait)
        self.wait = self.canvas.create_text(100, 530, anchor=W, font=('Arial', 14),
                                                 text='Wait time: ' + str(int(wait_time)))

        self.canvas.delete(self.through)
        self.through = self.canvas.create_text(100, 550, anchor=W, font=('Arial', 14),
                                                 text='Throughput: ' + str(int(throughput)))

        self.canvas.delete(self.sig_pattern)
        self.sig_pattern = self.canvas.create_text(100, 570, anchor=W, font=('Arial', 14),
                                                 text='Phase: ' + str(action))



        if(action == 0):
            self.canvas.itemconfig(self.cross1, fill='green')
            self.canvas.itemconfig(self.cross2, fill='red')
            self.canvas.itemconfig(self.cross3, fill='red')
            self.canvas.itemconfig(self.cross4, fill='red')
            self.canvas.itemconfig(self.cross5, fill='red')
            self.canvas.itemconfig(self.cross6, fill='green')
            self.canvas.itemconfig(self.cross7, fill='red')
            self.canvas.itemconfig(self.cross8, fill='red')

        if(action == 1):
            self.canvas.itemconfig(self.cross1, fill='red')
            self.canvas.itemconfig(self.cross2, fill='red')
            self.canvas.itemconfig(self.cross3, fill='red')
            self.canvas.itemconfig(self.cross4, fill='green')
            self.canvas.itemconfig(self.cross5, fill='red')
            self.canvas.itemconfig(self.cross6, fill='red')
            self.canvas.itemconfig(self.cross7, fill='green')
            self.canvas.itemconfig(self.cross8, fill='red')


        if(action == 2):
            self.canvas.itemconfig(self.cross1, fill='red')
            self.canvas.itemconfig(self.cross2, fill='green')
            self.canvas.itemconfig(self.cross3, fill='red')
            self.canvas.itemconfig(self.cross4, fill='red')
            self.canvas.itemconfig(self.cross5, fill='green')
            self.canvas.itemconfig(self.cross6, fill='red')
            self.canvas.itemconfig(self.cross7, fill='red')
            self.canvas.itemconfig(self.cross8, fill='red')

        if(action == 3):
            self.canvas.itemconfig(self.cross1, fill='red')
            self.canvas.itemconfig(self.cross2, fill='red')
            self.canvas.itemconfig(self.cross3, fill='green')
            self.canvas.itemconfig(self.cross4, fill='red')
            self.canvas.itemconfig(self.cross5, fill='red')
            self.canvas.itemconfig(self.cross6, fill='red')
            self.canvas.itemconfig(self.cross7, fill='red')
            self.canvas.itemconfig(self.cross8, fill='green')

    def vehicle_lane(self, lane_que, direc, location):


        if(direc == 'S' and location == 'W'):
            current_length = self.rd_1_lane_1_start_x
            for i in range(len(self.vehicle_blocks_W)):
                var = self.vehicle_blocks_W.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length - index.get_num_vehicles()
                self.vehicle_blocks_W.append(self.canvas.create_rectangle(end,  self.rd_1_lane_1_end_y, current_length, self.rd_1_lane_1_start_y,
                                                                          fill=self.mapColor(index.waiting_time()/max(1,index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end

        if (direc == 'R' and location == 'W'):
            current_length = self.rd_1_lane_2_start_x
            for i in range(len(self.vehicle_blocks_W_R)):
                var = self.vehicle_blocks_W_R.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length - index.get_num_vehicles()
                self.vehicle_blocks_W_R.append(
                    self.canvas.create_rectangle(end, self.rd_1_lane_2_end_y, current_length, self.rd_1_lane_2_start_y,
                                                 fill=self.mapColor(
                                                     index.waiting_time() / max(1, index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end

        if (direc == 'S' and location == 'N'):
            current_length = self.rd_2_lane_1_start_y
            for i in range(len(self.vehicle_blocks_N)):
                var = self.vehicle_blocks_N.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length - index.get_num_vehicles()
                self.vehicle_blocks_N.append(
                    self.canvas.create_rectangle(self.rd_2_lane_1_end_x, end, self.rd_2_lane_1_start_x, current_length,
                                                 fill=self.mapColor(
                                                     index.waiting_time() / max(1, index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end

        if (direc == 'R' and location == 'N'):
            current_length = self.rd_2_lane_2_start_y
            for i in range(len(self.vehicle_blocks_N_R)):
                var = self.vehicle_blocks_N_R.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length - index.get_num_vehicles()
                self.vehicle_blocks_N_R.append(
                    self.canvas.create_rectangle(self.rd_2_lane_2_end_x, end, self.rd_2_lane_2_start_x, current_length,
                                                 fill=self.mapColor(
                                                     index.waiting_time() / max(1, index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end

        if(direc == 'S' and location == 'E'):
            current_length = self.rd_3_lane_1_start_x
            for i in range(len(self.vehicle_blocks_E)):
                var = self.vehicle_blocks_E.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length + index.get_num_vehicles()
                self.vehicle_blocks_E.append(self.canvas.create_rectangle(end,  self.rd_3_lane_1_end_y, current_length, self.rd_3_lane_1_start_y,
                                                                          fill=self.mapColor(index.waiting_time()/max(1,index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end

        if (direc == 'R' and location == 'E'):
            current_length = self.rd_3_lane_2_start_x
            for i in range(len(self.vehicle_blocks_E_R)):
                var = self.vehicle_blocks_E_R.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length + index.get_num_vehicles()
                self.vehicle_blocks_W_R.append(
                    self.canvas.create_rectangle(end, self.rd_3_lane_2_end_y, current_length, self.rd_3_lane_2_start_y,
                                                 fill=self.mapColor(
                                                     index.waiting_time() / max(1, index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end
        if (direc == 'S' and location == 'S'):
            current_length = self.rd_4_lane_1_start_y
            for i in range(len(self.vehicle_blocks_S)):
                var = self.vehicle_blocks_S.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length + index.get_num_vehicles()
                self.vehicle_blocks_S.append(
                    self.canvas.create_rectangle(self.rd_4_lane_1_end_x, end, self.rd_4_lane_1_start_x, current_length,
                                                 fill=self.mapColor(
                                                     index.waiting_time() / max(1, index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end

        if (direc == 'R' and location == 'S'):
            current_length = self.rd_4_lane_2_start_y
            for i in range(len(self.vehicle_blocks_S_R)):
                var = self.vehicle_blocks_S_R.popleft()
                self.canvas.delete(var)

            for index in lane_que:
                end = current_length + index.get_num_vehicles()
                self.vehicle_blocks_S_R.append(
                    self.canvas.create_rectangle(self.rd_4_lane_2_end_x, end, self.rd_4_lane_2_start_x, current_length,
                                                 fill=self.mapColor(
                                                     index.waiting_time() / max(1, index.get_num_vehicles()))))
                # self.canvas.coords(self.rd_1_lane_1, self.rd_1_lane_1_end_x, self.rd_1_lane_1_end_y,self.rd_1_lane_1_start_x,self.rd_1_lane_1_start_y)
                current_length = end
        self.canvas.after(5)
        self.canvas.update()

    def mapColor(self, val):
        if val < 10:
            return '#0f0'
        elif val < 20:
            return '#1a0'
        elif val < 30:
            return '#380'
        elif val < 40:
            return '#ff0'
        elif val < 50:
            return '#ea0'
        elif val < 60:
            return '#f81'
        elif val < 70:
            return '#f50'
        elif val < 80:
            return '#e20'
        elif val < 90:
            return '#d10'
        else:
            return '#f00'