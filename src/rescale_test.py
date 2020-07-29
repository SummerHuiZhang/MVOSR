# recover ego-motion scale by road geometry information
# @author xiangwei(wangxiangwei.cpp@gmail.com) and zhanghui()
# 

# @function: calcualte the road pitch angle from motion matrix
# @input: the tansformation matrix in SE(3) 
# @output:translation angle calculate by t in R^3 and 
# rotarion angle calculate from rotation matrix R in SO(3)



from scipy.spatial import Delaunay
from estimate_road_norm import *
import numpy as np
import math
from collections import deque 
import matplotlib.pyplot as plt
import scale_calculator as sc
import graph 
import param

class ScaleEstimator:
    def __init__(self,absolute_reference,window_size=6):
        self.absolute_reference = absolute_reference
        self.camera_pitch       = 0
        self.scale  = 1
        self.inliers = None
        self.scale_queue = deque()
        self.window_size = window_size
        self.vanish =  185
        self.sc = sc.ScaleEstimator(absolute_reference,window_size)
        self.gc = graph.GraphChecker([[3,1],[2,2],[2,2],[0,4]])
        self.img_w =  param.img_w
        self.img_h =  param.img_h
    def initial_estimation(self,motion_matrix):

        return 0

    '''
    check the three vertice in triangle whether satisfy
    (d_1-d_2)*(v_1-v_2)<=0 if not they are outlier
    True means outlier
    '''
    def check_triangle(self,v,d):
        flag=[False,False,False]
        a = (v[0]-v[1])*(d[0]-d[1])
        b = (v[0]-v[2])*(d[0]-d[2])
        c = (v[1]-v[2])*(d[1]-d[2])
        if a>0:
            flag[0]=True
            flag[1]=True
        if b>0:
            flag[0]=True
            flag[1]=True
        if c>0:
            flag[1]=True
            flag[2]=True
        return flag

    def find_outliers(self,feature3d,feature2d,triangle_ids):
        # suppose every is inlier
        outliers = np.ones((feature3d.shape[0]))
        for triangle_id in triangle_ids:
            data=[]
            depths   = feature3d[triangle_id,2]
            pixel_vs = feature2d[triangle_id,1]
            flag    = self.check_triangle(pixel_vs,depths)
            outlier = triangle_id[flag] 
            outliers[outlier]-=np.ones(outliers[outlier].shape[0])
        
        return outliers
    
    def feature_selection(self,feature3d,feature2d):
        lower_feature_ids = feature2d[:,1]>self.vanish
        feature2d = feature2d[lower_feature_ids,:]
        feature3d = feature3d[lower_feature_ids,:]
        tri = Delaunay(feature2d)
        triangle_ids = tri.simplices
        '''
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        valid_id_1     = self.sc.find_reliability_by_graph(feature3d,feature2d,triangle_ids)
        valid_id_2  = self.gc.find_inliers(feature3d,feature2d,triangle_ids)
        outliers = self.find_outliers(feature3d,feature2d,triangle_ids)
        valid_id_3 = (outliers>=0)
        valid_id = valid_id_1
        #valid_id =[]
        ax1.plot(feature3d[:,2],-feature3d[:,1],'.r')
        ax2.plot(feature3d[:,2],-feature3d[:,1],'.r')
        ax3.plot(feature3d[:,2],-feature3d[:,1],'.r')
        ax1.plot(feature3d[valid_id_1,2],-feature3d[valid_id_1,1],'.g')
        ax2.plot(feature3d[valid_id_2,2],-feature3d[valid_id_2,1],'.g')
        ax3.plot(feature3d[valid_id_3,2],-feature3d[valid_id_3,1],'.g')
        plt.show()
        '''
        #valid_id     = self.sc.find_reliability_by_graph(feature3d,feature2d,triangle_ids)
        valid_id  = self.gc.find_inliers(feature3d,feature2d,triangle_ids)
        #outliers = self.find_outliers(feature3d,feature2d,triangle_ids)
        #valid_id = (outliers>=0)
        #valid_id = []
        print('feature rejected ',np.sum(valid_id==False))
        print('feature left     ',np.sum(valid_id))
        if(np.sum(valid_id)>10):
            feature2d = feature2d[valid_id,:]
            feature3d = feature3d[valid_id,:]
            tri = Delaunay(feature2d)
            triangle_ids = tri.simplices

        b_matrix = np.matrix(np.ones((3,1),np.float))
        data = []
        #calculating the geometry model of each triangle
        for triangle_id in triangle_ids:
            point_selected = feature3d[triangle_id]
            a_array = np.array(point_selected)
            a_matrix = np.matrix(a_array)
            a_matrix_inv = a_matrix.I
            norm = a_matrix_inv*b_matrix
            norm_norm_2 = norm.T*norm#the square norm of norm
            height = 1/math.sqrt(norm_norm_2)
            if norm[1,0]<0:
                norm = -norm
                height = -height
            #height= np.mean(point_selected[:,1])
            pitch = math.asin(-norm[1,0]/math.sqrt(norm_norm_2[0,0]))
            pitch_deg = pitch*180/3.1415926
            pitch_height = [norm[1,0]/math.sqrt(norm_norm_2[0,0]),pitch_deg,height]
            data.append(pitch_height)
        data = np.array(data) # all data is saved here

        # initial select by prior information
        data_sub = data[data[:,1]>self.camera_pitch-95]#>80 deg
        data_sub = data_sub[data[:,1]<self.camera_pitch-80]#>80 deg
        data_sub = data_sub[data_sub[:,2]>0]#under
        precomput_h = 0.9*np.median(data_sub[:,2])
        print(precomput_h)
        #unvalid_id = data[:,1]>=self.camera_pitch-80
        #precomput_h  = np.mean(data[unvalid_id,2])
        data_sub_low = data_sub[data_sub[:,2]>precomput_h,:]
        data_id = []
        # collect suitable points and triangle
        for i in range(0,triangle_ids.shape[0]):
            triangle_id = triangle_ids[i]
            pitch_deg = data[i,1]
            height = data[i,2]
            triangle_points =  np.array(feature2d[triangle_id],np.int32)
            if(pitch_deg>self.camera_pitch-95 and pitch_deg<self.camera_pitch-85):
                #if(height>precomput_h):
                #if(True):
                if(height>precomput_h):
                        data_id.append(triangle_id[0])
                        data_id.append(triangle_id[1])
                        data_id.append(triangle_id[2])
                        pts = triangle_points.reshape((-1,1,2))
        #data_id = list(np.unique(data_id))
        
        #data_id = (feature2d[:,0]>self.img_w*0.4) & (feature2d[:,0]<self.img_w*0.6)
        #data_id &= (feature2d[:,1]>self.img_h*0.666)
        point_selected = feature3d[data_id]
        '''
        ax1 = plt.subplot(111)
        ax1.plot(feature3d[:,2],-feature3d[:,1],'.g')
        ax1.plot(point_selected[:,2],-point_selected[:,1],'.y')
        plt.show()
        '''
        return point_selected
        #return feature3d
    def scale_calculation_ransac(self,point_selected):
        if(point_selected.shape[0]>=12):
            # ransac
            a_array = np.array(point_selected)
            m,b = get_pitch_ransac(a_array,100,0.005)
            road_model_ransac = np.matrix(m)
            norm = road_model_ransac[0,0:-1]
            h_bar = -road_model_ransac[0,-1]
            if norm[0,1]<0:
                norm = -norm
                h_bar = -h_bar
            norm_norm_2 = norm*norm.T
            norm_norm = math.sqrt(norm_norm_2)/h_bar
            norm = norm/h_bar
            ransac_camera_height = 1/norm_norm
            pitch = math.asin(norm[0,1]/norm_norm)
            scale = self.absolute_reference/ransac_camera_height
#0.3 is the max accellerate
            if scale - self.scale >0.3:
                self.scale += 0.3 
            elif scale - self.scale<-0.3:
                self.scale -= 0.3
            else:
                self.scale = scale
        self.scale_queue.append(self.scale)
        if len(self.scale_queue)>self.window_size:
            self.scale_queue.popleft()
        return np.median(self.scale_queue),1



    def scale_calculation(self,feature3d,feature2d,img=None):
        point_selected = self.feature_selection(feature3d,feature2d)
        return self.scale_calculation_ransac(point_selected)
        #return self.sc.scale_calculation_static(point_selected)
        # initial ransac
def main():
    # get initial motion pitch by motion matrix
    # triangle region norm and height
    # selected
    # calculate the road norm
    # filtering
    # scale recovery
    camera_height = 1.7
    scale_estimator = ScaleEstimator(camera_height)
    scale = scale_estimator.scale_calculation()

if __name__ == '__main__':
    main()
