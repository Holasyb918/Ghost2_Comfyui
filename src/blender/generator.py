from .fpn import FPN
from .blender_3p import Decoder
from torch import nn
import torch
import torch.nn.functional as F

class BlenderGenerator(nn.Module):
    def __init__(self, decoder_ic=12, dilate_kernel=17, out_layer=18, f_in_channels=256, f_inter_channels=256, temperature=0.001):
        super().__init__()
        self.feature_ext = FPN(out_channels=f_inter_channels, out_layer=out_layer)
        self.decoder = Decoder(ic=decoder_ic)
        self.dilate = nn.MaxPool2d(kernel_size=dilate_kernel, 
                        stride=1, 
                        padding=dilate_kernel//2)

        # self.phi = nn.Conv2d(in_channels=f_in_channels, 
        #                     out_channels=f_inter_channels, kernel_size=1, stride=1, padding=0)
        # self.theta = nn.Conv2d(in_channels=f_in_channels, 
        #                     out_channels=f_inter_channels, kernel_size=1, stride=1, padding=0)

        self.temperature = temperature

        self.head_index = [2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20] #[1,2,3,4,5,6,7,8,9,10,11,12,13,17,18]
        self.eps = 1e-8
        
        self.threshold_dict = {
                                'skin': 3000, 
                                'hair': 6000,
                                'eyes': 150,
                                'brows': 200,
                                'ears': 280,
                                'nose': 800,
                                'lips': 730, 
                                'teeth': 250,
                                'glasses': 25,
                                'hat': 6000, 
                                'headphone': 200,
                                'earrings': 20,
                                'beard': 900,
                                'ear_left': 100,
                                'ear_right': 100
                                }

    def forward(
        self, I_a, I_gray, I_t, M_a, M_t,
        gt=None, cycle=False, train=False, return_inputs=False,
        M_a_noise=None, M_t_noise=None, old_version=False,
        copy_source_attrb=False, inpainter=None
    ):

        fA = self.feature_ext(I_a)
        fT = self.feature_ext(I_t)
        
        # fA = self.phi(fA)
        # fT = self.theta(fT)
        
        
        gen_h, gen_i, mask_list, matrix_list, M_A_resize_list, fAr_list, fTr_list, fAr_mask_list, fTr_mask_list, ref_list =\
            self.RCNet(fA,fT,M_a,M_t,I_t,M_a_noise,M_t_noise, I_gray, I_a, old_version=old_version, copy_source_attrb=copy_source_attrb)
        
        M_Ah,M_Ad,M_Td,M_Ai,M_Ti,M_Ar,M_Tr = mask_list
        
        gen_total = gen_h + gen_i
        I_gd = F.interpolate(I_gray * M_Td, size=gen_total.shape[-2:], mode='bilinear', align_corners=True)
        
        gen_h = F.interpolate(gen_h,size=I_t.shape[-2:],mode='bilinear',align_corners=True)
        gen_i = F.interpolate(gen_i,size=I_t.shape[-2:],mode='bilinear',align_corners=True)

        

        if cycle:
            cycle_gen = self.RCCycle(
                gen_h+gen_i,
                [M_Ar,M_Tr,M_Ai,M_Ti],
                matrix_list, M_A_resize_list, fAr_mask_list, fTr_mask_list, ref_list,
                fA.shape
            )

            I_td = I_t * M_Td
            I_td = F.interpolate(I_td, size=cycle_gen.shape[-2:],mode='bilinear')
            return cycle_gen, I_td

        I_tb = gt * (1-M_Ad)
        I_ag = I_gray * M_Ah
        if inpainter is not None:
            gen_i = inpainter(gen_h, I_tb, M_Ai, M_Ah, I_a)
        else:
            gen_i = I_t * M_Ai
        
        M_Ah = self.get_mask(M_a,self.head_index)
        M_Th = self.get_mask(M_t,self.head_index)
        M_sum = M_Ah + M_Th + (0 if M_a_noise is None else M_a_noise) + (0 if M_t_noise is None else M_t_noise)
        M_Ai,M_Ad = self.get_inpainting(M_sum,M_Ah)
        
        inp = torch.cat([gen_h,gen_i,
                         M_Ah,
                         I_tb,M_Ai,I_ag],1)
        
        oup = self.decoder(inp)
        
        if return_inputs:
            return oup, gen_h, gen_i, M_Ah, I_tb, M_Ai, I_ag

        if train:
            return oup, M_Ah, M_Ai, gen_total, I_gd

        return oup
       

    def RCNet(
        self, fA, fT, M_a, M_t, I_t,
        M_a_noise=None, M_t_noise=None,
        I_gray=None, I_a=None,
        old_version=False, copy_source_attrb=False
    ):
        M_Ah = self.get_mask(M_a,self.head_index)
        M_Th = self.get_mask(M_t,self.head_index)

        M_Ti,M_Td = self.get_inpainting(M_Th)
        M_sum = M_Ah + M_Th + (0 if M_a_noise is None else M_a_noise) + (0 if M_t_noise is None else M_t_noise)
        M_Ai,M_Ad = self.get_inpainting(M_sum,M_Ah)
        M_Ar = self.get_multi_mask(M_a, old_version=old_version)
        M_Tr = self.get_multi_mask(M_t, old_version=old_version)
        
        matrix_list = []
        M_A_resize_list = []
        fAr_list = []
        fTr_list = []
        fAr_mask_list = []
        fTr_mask_list = []
        ref_list = []
        
        gen_h = torch.zeros((I_a.size(0), 3, 64, 64)).to(I_a.device)
        copy_eyes_glasses = torch.zeros(I_a.size(0)).to(torch.bool).to(I_a.device)
        
        for (key_a, m_a),(key_t, m_t) in zip(M_Ar.items(),M_Tr.items()): #
            
            if old_version:
                matrix, M_A_resize, fAr, fTr, fAr_mask, fTr_mask, ref = self.compute_corre_and_masks(fA, fT, m_a, m_t, I_t)
                gen_h = self.compute_reference(matrix, M_A_resize, fAr_mask, fTr_mask, ref, gen=gen_h)
                
            else: #with improved inpainting
                source_inpaint = torch.zeros(I_a.size(0)).to(torch.bool).to(I_a.device)
                missing_mask = torch.sum(m_t, dim=(1, 2, 3))
                missing_mask_bool = missing_mask < self.threshold_dict[key_a] #find batches with missing mask regions in target
                
                #refine mask for glasses if present
                if key_a == 'glasses':
                    missing_mask_bool_pos = ~missing_mask_bool #batches where glasses are present
                    if torch.sum(missing_mask_bool_pos) > 0: #any target image has glasses
                        #compute new masks for glasses and skin for target
                        m_t[missing_mask_bool_pos], M_Tr['skin'][missing_mask_bool_pos] =\
                            self.close_glasses_mask(m_t[missing_mask_bool_pos],  M_Tr['skin'][missing_mask_bool_pos]) 
                        check_ma_mask = torch.sum(m_a[missing_mask_bool_pos], dim=(1,2,3)) #check if corresponding source does not have glasses
                        copy_eyes_glasses[missing_mask_bool_pos] = check_ma_mask < self.threshold_dict[key_a] #copy eyes from such source images
                
                if torch.sum(missing_mask_bool) > 0: # if any instance has missing mask
                    check_ma_mask = torch.zeros(I_a.size(0)).to(I_a.device)
                    check_ma_mask[missing_mask_bool] = torch.sum(m_a[missing_mask_bool], dim=(1,2,3))
                    check_ma_mask = check_ma_mask > 0 #check that area missing in target is present in source
                    
                    if torch.sum(check_ma_mask) > 0: #if not present in target, but present in source
                        if key_a == 'ear_left' or key_a == 'ear_right' or key_a == 'nose':
                            m_t[missing_mask_bool & check_ma_mask] = M_Tr['skin'][missing_mask_bool & check_ma_mask]

                        if key_a == 'brows':
                            m_t, source_inpaint = self.check_hair_brows_beard('hair', 'beard', m_t, M_Tr, missing_mask_bool & check_ma_mask, copy_source_attrb=copy_source_attrb)   
                                    
                        if key_a == 'beard':
                            m_t[missing_mask_bool & check_ma_mask] = M_Tr['skin'][missing_mask_bool & check_ma_mask]
  
                                    
                        if key_a == 'hair':
                            m_t, source_inpaint = self.check_hair_brows_beard('beard', None, m_t, M_Tr, missing_mask_bool & check_ma_mask, copy_source_attrb=copy_source_attrb)

                        if key_a in ['teeth', 'glasses', 'hat', 'headphone', 'earrings', 'lips', 'eyes']: #use reference from source
                            source_inpaint = missing_mask_bool & check_ma_mask
                            
                            
                if torch.sum(copy_eyes_glasses) > 0 and key_a == 'eyes': #copy eyes from source if glasses are present in target
                    m_a[copy_eyes_glasses] = kornia_morphology.dilation(
                                             m_a[copy_eyes_glasses],
                                             kernel = self.get_circular_kernel(5).to(m_a.device)
                                        ) 
                    M_Ar['skin'][copy_eyes_glasses] = (
                        M_Ar['skin'][copy_eyes_glasses] -
                        ((m_a[copy_eyes_glasses] == 1) & (M_Ar['skin'][copy_eyes_glasses] == 1)
                    ).to(torch.int))
                    source_inpaint[copy_eyes_glasses] = True
                    
                matrix, M_A_resize, fAr, fTr, fAr_mask, fTr_mask, ref = self.compute_corre_and_masks(fA, fT, m_a, m_t, I_t)
                    
                if copy_source_attrb and key_a in [
                    'glasses', 'hat', 'headphone', 'earrings', 'hair', 'brows', 'lips'
                ]: 
                    source_inpaint[source_inpaint == False] = True
                
                if torch.sum(~source_inpaint) > 0: #recolor from target
                    gen_h[~source_inpaint] = self.compute_reference(
                        matrix[~source_inpaint], M_A_resize[~source_inpaint],
                        fAr_mask[~source_inpaint], fTr_mask[~source_inpaint],
                        ref[~source_inpaint], gen=gen_h[~source_inpaint]
                    )
                
                if torch.sum(source_inpaint) > 0: #copy from source
                    gen_h[source_inpaint] = self.compute_MA_reference(
                        I_a[source_inpaint], I_gray[source_inpaint],
                        I_t[source_inpaint], M_A_resize[source_inpaint],
                        m_a[source_inpaint], gen=gen_h[source_inpaint],
                        mask_face_a=M_Ar['skin'][source_inpaint], mask_face_t=M_Tr['skin'][source_inpaint],
                        mask_head_a=M_Ah[source_inpaint], mask_head_t=M_Th[source_inpaint]
                    )
           
            
            matrix_list.append(matrix)
            M_A_resize_list.append(M_A_resize)
            fAr_list.append(fAr)
            fTr_list.append(fTr)
            fAr_mask_list.append(fAr_mask)
            fTr_mask_list.append(fTr_mask)
            ref_list.append(ref)

        gen_i = None
        matrix, M_A_resize, fAr, fTr, fAr_mask, fTr_mask, ref = self.compute_corre_and_masks(
            fA, fT, M_Ai, M_Ti, I_t
        )
        gen_i = self.compute_reference(matrix, M_A_resize, fAr_mask, fTr_mask, ref, gen=gen_i)
        
        matrix_list.append(matrix)
        M_A_resize_list.append(M_A_resize)
        fAr_list.append(fAr)
        fTr_list.append(fTr)
        fAr_mask_list.append(fAr_mask)
        fTr_mask_list.append(fTr_mask)
        ref_list.append(ref)
        
        M_Tr = list(M_Tr.values()) #to remove dict format
        M_Ar = list(M_Ar.values())
        mask_list = [M_Ah, M_Ad, M_Td, M_Ai, M_Ti, M_Ar, M_Tr]
        
        return (
            gen_h,
            gen_i,
            mask_list,
            matrix_list,
            M_A_resize_list,
            fAr_list,
            fTr_list,
            fAr_mask_list,
            fTr_mask_list,
            ref_list
        )
    
    def check_hair_brows_beard(self, key_1, key_2, m_t, M_Tr, missing_mask_bool, copy_source_attrb=False):
        """ recolor according to key_1, if it is not present, recolor according to key_2, otherwise copy from sorce """
        if copy_source_attrb:
            source_inpaint = missing_mask_bool
            return m_t, source_inpaint
        else:
            source_inpaint = torch.zeros(m_t.size(0)).to(torch.bool)
            
            present_ind = self.check_if_missing(M_Tr[key_1], key_1)
            m_t[missing_mask_bool & present_ind] = M_Tr[key_1][missing_mask_bool & present_ind]

            if torch.sum((~present_ind) & missing_mask_bool) > 0: #no key_1 in target
                if key_2 is not None:
                    present_ind2 = self.check_if_missing(M_Tr[key_2], key_2)
                    m_t[present_ind2 & missing_mask_bool & (~present_ind)] = M_Tr[key_2][present_ind2 & missing_mask_bool & (~present_ind)]
                else:
                    present_ind2 = torch.zeros(m_t.size(0)).to(torch.bool).to(m_t.device)                    
                
                if  torch.sum((~present_ind2) & (~present_ind) & missing_mask_bool) > 0: #if not key_1 and key_2 in target
                    
                    source_inpaint = (~present_ind) & (~present_ind2) & missing_mask_bool

                    
                
            return m_t, source_inpaint.to(torch.bool)

    
    def check_if_missing(self, mask, key):
        """ find batch indices that contain mask for a given key """
        present_mask = torch.sum(mask, dim=(1, 2, 3))
        present_mask_bool = present_mask > self.threshold_dict[key]
        return present_mask_bool
    
    def compute_MA_reference(self, I_a, I_gray, I_t, M_A_resize, m_a, gen=None, mask_face_a=None, mask_face_t=None, mask_head_a=None, mask_head_t=None, recolor_eyes=False):
        """ copy region from source according to M_A_resize """
        ref = I_a
        
        if recolor_eyes:
            matched_image = color_transfer((I_a[:, [2, 1, 0], ...] / 2 + 0.5), (I_t[:, [2, 1, 0], ...] / 2 + 0.5) * mask_head_t, mode='pca')
            ref = (matched_image[:, [2, 1, 0], ...] - 0.5) * 2
        
        ref = F.interpolate(ref, size=gen.size()[-2:], mode='bilinear', align_corners=True)
        
        gen = gen + ref * M_A_resize
        
        return gen
    
                        
    def get_circular_kernel(self, diameter):
            mid = (diameter - 1) / 2
            distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
            kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

            return torch.from_numpy(kernel)    
                        
    def close_glasses_mask(self, mask_glass, mask_skin, kernel_size=61):
        """ perform closing for mask of glasses """
        mask_glass = kornia_morphology.closing(
            mask_glass,
            kernel = self.get_circular_kernel(kernel_size).to(mask_glass.device)
        ) 
        
        mask_skin = mask_skin - ((mask_glass == 1) & (mask_skin == 1)).to(torch.int)
                
        
        return mask_glass, mask_skin
        

    def RCCycle(
        self, I_t,
        mask_list, matrix_list, M_A_resize_list, fAr_mask_list, fTr_mask_list, ref_list,
        shape
    ):
        M_Ar,M_Tr,M_Ai,M_Ti = mask_list
        batch,channel,h,w = shape

        gen_h = None 
        for matrix, M_A_resize, fAr_mask, fTr_mask, ref in zip(
            matrix_list[:-1],
            M_A_resize_list,
            fAr_mask_list,
            fTr_mask_list,
            ref_list,
        ):
            gen_h = self.compute_reference(matrix, M_A_resize, fAr_mask, fTr_mask, ref, gen=gen_h)
        
        matrix, M_A_resize, fAr_mask, fTr_mask, ref = (
            matrix_list[-1], M_A_resize_list[-1], fAr_mask_list[-1],
            fTr_mask_list[-1], ref_list[-1]
        )
        
        gen_i = self.compute_reference(matrix, M_A_resize, fAr_mask, fTr_mask, ref, gen=None)

        return gen_h + gen_i

    def get_inpainting(self,M,head=None, and_mask=None):
        M = torch.clamp(M,0,1)
        M_dilate = self.dilate(M)
        if head is None:
            MI = M_dilate - M
        else:
            MI = M_dilate - head
            
        if and_mask is not None:
            MI = torch.where(and_mask > 0, MI, 0)
            
        return MI,M_dilate
    
    def get_all_mask(self, M_a):
        return self.get_mask(M_a, list(range(1, 19)))
    
    def get_multi_mask(self,M_a, old_version=False):
        if old_version:
            # skin
            skin_mask_A = self.get_mask(M_a, [2, 14, 18]) #[1])
            # hair 
            hair_mask_A = self.get_mask(M_a, [15, 16, 17, 20]) #[17,18])

            # eye 
            eye_mask_A = self.get_mask(M_a, [5, 6]) #[4,5,6])

            # brow
            brow_mask_A = self.get_mask(M_a, [3, 4]) #[2,3])

            # ear 
            ear_mask_A = self.get_mask(M_a, [10, 11, 19]) #[7,8,9])

            #nose
            nose_mask_A = self.get_mask(M_a, [12]) #[10])

            # lip
            lip_mask_A = self.get_mask(M_a, [9])#[12,13])


            # tooth
            tooth_mask_A = self.get_mask(M_a, [7, 8])#[11])

            return {
                'skin':skin_mask_A, 
                'hair':hair_mask_A,
                'eyes':eye_mask_A,
                'brows':brow_mask_A,
                'ears':ear_mask_A,
                'nose':nose_mask_A,
                'lips':lip_mask_A, 
                'teeth':tooth_mask_A
            }
        else:
            # skin
            beard_mask_A = self.get_mask(M_a, [14])
            
            skin_mask_A = self.get_mask(M_a, [2])
            
            # hair 
            hair_mask_A = self.get_mask(M_a, [15, 20])

            # eye 
            eye_mask_A = self.get_mask(M_a, [5, 6])

            # brow
            brow_mask_A = self.get_mask(M_a, [3, 4])

            # ear 
            ear_mask_A_left = self.get_mask(M_a, [10])
            
            ear_mask_A_right = self.get_mask(M_a, [11])
            
            #nose
            nose_mask_A = self.get_mask(M_a, [12])

            # lip
            lip_mask_A = self.get_mask(M_a, [9])

            glasses_mask_A = self.get_mask(M_a, [18])

            hat_mask_A = self.get_mask(M_a, [16])

            headphone_mask_A = self.get_mask(M_a, [17])

            earrings_mask_A = self.get_mask(M_a, [19])
            
            # tooth
            tooth_mask_A = self.get_mask(M_a, [7, 8])

            return {
                    'hair':hair_mask_A,
                    'glasses':glasses_mask_A,
                    'eyes':eye_mask_A,
                    'skin':skin_mask_A,
                    'brows':brow_mask_A,
                    'nose':nose_mask_A,
                    'lips':lip_mask_A, 
                    'teeth':tooth_mask_A,
                    'hat':hat_mask_A,
                    'headphone':headphone_mask_A,
                    'earrings':earrings_mask_A,
                    'beard': beard_mask_A,
                    'ear_left': ear_mask_A_left,
                    'ear_right': ear_mask_A_right,
                    }

    def get_mask(self, mask, indexs):
        out = torch.zeros_like(mask, device=next(self.parameters()).device)
        for i in indexs:
            out[mask == i] = 1

        return out

    def normlize(self,x):
        x_mean = x.mean(dim=1,keepdim=True)
        x_norm = torch.norm(x,2,1,keepdim=True) + self.eps 
        return (x-x_mean) / x_norm
    
    def transform_fX(self, fX, M_X_resize, I_X_resize=None):
        batch,channel,h,w = fX.shape
        
        nonzeros_cnt_X = (M_X_resize == 1).sum(dim=(1, 2, 3))
        max_nonzeros_cnt_X = nonzeros_cnt_X.max().item()
        fXr_mask = (
            torch.arange(max_nonzeros_cnt_X, device=fX.device)[None, :] < nonzeros_cnt_X[:, None]
        )

        fXr = torch.zeros(batch, channel, max_nonzeros_cnt_X, dtype=fX.dtype, device=fX.device).masked_scatter(
            fXr_mask[:, None, :], fX.masked_select(M_X_resize == 1)
        )

        if I_X_resize is not None:
            ref = torch.zeros(batch, I_X_resize.shape[1], max_nonzeros_cnt_X, dtype=fX.dtype, device=fX.device).masked_scatter(
                fXr_mask[:, None, :], I_X_resize.masked_select(M_X_resize == 1).to(fX.dtype)
            )

            return fXr, fXr_mask, ref

        return fXr, fXr_mask

    def compute_corre_and_masks(self, fA, fT, M_A, M_T, I_t):
        batch,channel,h,w = fA.shape
        
        M_A_resize = F.interpolate(M_A, size=(h,w),mode='nearest')
        M_T_resize = F.interpolate(M_T, size=(h,w),mode='nearest')
        I_t_resize = F.interpolate(I_t, size=(h,w),mode='bilinear', align_corners=True)

        fAr, fAr_mask = self.transform_fX(fA, M_A_resize) # [b, c, hA]; [b, hA]
        fTr, fTr_mask, ref = self.transform_fX(fT, M_T_resize, I_t_resize) # [b, c, hT]; [b, hT]; [b, 3, hT]

        fAr = self.normlize(fAr)
        fTr = self.normlize(fTr)
        
        matrix = torch.matmul(fAr.permute(0, 2, 1), fTr) # [b, hA, hT]
        
        return matrix, M_A_resize, fAr, fTr, fAr_mask, fTr_mask, ref

    
    def compute_reference(self, matrix, M_A_resize, fAr_mask, fTr_mask, ref, gen=None):
        batch, _, h, w = M_A_resize.shape
        
        if gen is None:
            gen = torch.zeros((batch,3,h,w), device=matrix.device, dtype=matrix.dtype)

        f_WTA = matrix / self.temperature

        neg_inf = -(10 ** 4) # float('-inf')
        f_WTA[(~fAr_mask[:, :, None]).expand_as(f_WTA)] = neg_inf
        f_WTA[(~fTr_mask[:, None, :]).expand_as(f_WTA)] = neg_inf
        f = F.softmax(f_WTA, dim=-1, dtype=torch.float32) # [b, hA, hT]

        ref = torch.matmul(ref, f.permute(0, 2, 1)) # [b, 3, hT] x [b, hT, hA] = [b, 3, hA]

        gen = gen.to(ref.dtype)
                
        gen = gen.masked_scatter(
            M_A_resize == 1,
            ref.masked_select(fAr_mask[:, None, :].expand_as(ref))
        )

        return gen
