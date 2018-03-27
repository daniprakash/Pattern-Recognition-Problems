
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


clap = cv2.imread('1_orange.png', 0)


# In[34]:


clap


# In[41]:


clap.shape


# In[40]:


plt.imshow(clap)output = greycomatrix(clap, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[ ]:


output = greycomatrix(clap, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[45]:


contrast_clap = greycoprops(output, 'contrast')
dissimilarity_clap = greycoprops(output, 'dissimilarity')
homogeneity_clap = greycoprops(output, 'homogeneity')
ASM_clap = greycoprops(output, 'ASM')
energy_clap = greycoprops(output, 'energy')
result[:, :, 0, 1]


# In[55]:


clap2 = cv2.imread('2_grass.png', 0)


# In[56]:


clap2.shape


# In[57]:


clap2


# In[ ]:


output1 = greycomatrix(clap2, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[65]:


output1[:, :, 0, 1][64,84]


# In[68]:


contrast_clap2 = greycoprops(output1, 'contrast')
dissimilarity_clap2 = greycoprops(output1, 'dissimilarity')
homogeneity_clap2 = greycoprops(output1, 'homogeneity')
ASM_clap2 = greycoprops(output1, 'ASM')
energy_clap2 = greycoprops(output1, 'energy')


# In[74]:


clap3 = cv2.imread('3_classify_green.png', 0)


# In[75]:


clap3.shape


# In[80]:


clap3


# In[83]:


output2 = greycomatrix(clap3, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[87]:


output2[127,149, 0, 1]


# In[86]:


output2.shape


# In[88]:


contrast_clap3 = greycoprops(output2, 'contrast')
dissimilarity_clap3 = greycoprops(output2, 'dissimilarity')
homogeneity_clap3 = greycoprops(output2, 'homogeneity')
ASM_clap3 = greycoprops(output2, 'ASM')
energy_clap3 = greycoprops(output2, 'energy')


# In[96]:


plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(clap)
plt.title('class1')

plt.subplot(142),plt.imshow(contrast_clap)
plt.title('contrast_clap')

plt.subplot(143),plt.imshow(dissimilarity_clap)
plt.title('dissimilarity_clap')

plt.subplot(144),plt.imshow(homogeneity_clap)
plt.title('homogenity_clap')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_clap)
plt.title('ASM_clap')

plt.subplot(122),plt.imshow(energy_clap)
plt.title('energy_clap')

plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(clap2)
plt.title('class2')

plt.subplot(142),plt.imshow(contrast_clap2)
plt.title('contrast_clap2')

plt.subplot(143),plt.imshow(dissimilarity_clap2)
plt.title('dissimilarity_clap2')

plt.subplot(144),plt.imshow(homogeneity_clap2)
plt.title('homogenity_clap2')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_clap2)
plt.title('ASM_clap2')

plt.subplot(122),plt.imshow(energy_clap2)
plt.title('energy_clap2')


plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(clap3)
plt.title('sample image')

plt.subplot(142),plt.imshow(contrast_clap3)
plt.title('contrast_clap3')

plt.subplot(143),plt.imshow(dissimilarity_clap3)
plt.title('dissimilarity_clap3')

plt.subplot(144),plt.imshow(homogeneity_clap3)
plt.title('homogenity_clap3')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_clap3)
plt.title('ASM_clap3')

plt.subplot(122),plt.imshow(energy_clap3)
plt.title('energy_clap3')

plt.show()


# In[97]:


plt.figure(figsize=(15,8))

plt.subplot(131),plt.scatter(contrast_clap, dissimilarity_clap)
plt.title('contrast_clap vs dissimilarity_clap')

plt.subplot(132),plt.scatter(contrast_clap2, dissimilarity_clap2)
plt.title('contrast_clap2 vs dissimilarity_clap2')

plt.subplot(133),plt.scatter(contrast_clap3, dissimilarity_clap3)
plt.title('contrast_clap3 vs dissimilarity_clap3')

plt.show()


# In[95]:


plt.figure(figsize=(12,6))

plt.subplot(141),plt.imshow(output[:, :, 0, 0])
plt.title('clap 0 angle')

plt.subplot(142),plt.imshow(output[:, :, 0, 1])
plt.title('clap 45 angle')

plt.subplot(143),plt.imshow(output[:, :, 0, 2])
plt.title('clap 90 angle')

plt.subplot(144),plt.imshow(output[:, :, 0, 3])
plt.title('clap 135 angle')

plt.figure(figsize=(12,6))

plt.subplot(141),plt.imshow(output1[:, :, 0, 0])
plt.title('clap2 0 angle')

plt.subplot(142),plt.imshow(output1[:, :, 0, 1])
plt.title('clap2 45 angle')

plt.subplot(143),plt.imshow(output1[:, :, 0, 2])
plt.title('clap2 90 angle')

plt.subplot(144),plt.imshow(output1[:, :, 0, 3])
plt.title('clap2 135 angle')

plt.figure(figsize=(12,6))

plt.subplot(141),plt.imshow(output2[:, :, 0, 0])
plt.title('clap3 0 angle')

plt.subplot(142),plt.imshow(output2[:, :, 0, 1])
plt.title('clap3 45 angle')

plt.subplot(143),plt.imshow(output2[:, :, 0, 2])
plt.title('clap3 90 angle')

plt.subplot(144),plt.imshow(output2[:, :, 0, 3])
plt.title('clap3 135 angle')

plt.show()

