
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg


# In[27]:


original_pic = cv2.imread('original_pic.png', 0)
picture_1 = cv2.imread('picture_1.png',0)
picture_2 = cv2.imread('picture_2.png', 0)


# In[28]:


original_pic


# In[29]:


original_pic.shape


# In[54]:


plt.imshow(original_pic)


# In[51]:


result = greycomatrix(original_pic, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[31]:


contrast_original_pic = greycoprops(result, 'contrast')
dissimilarity_original_pic = greycoprops(result, 'dissimilarity')
homogeneity_original_pic = greycoprops(result, 'homogeneity')
ASM_original_pic = greycoprops(result, 'ASM')
energy_original_pic = greycoprops(result, 'energy')


# In[32]:


plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(original_pic)
plt.title('original_pic')

plt.subplot(142),plt.imshow(contrast_original_pic)
plt.title('contrast_original_pic')

plt.subplot(143),plt.imshow(dissimilarity_original_pic)
plt.title('dissimilarity_original_pic')

plt.subplot(144),plt.imshow(homogeneity_original_pic)
plt.title('homogenity_original_pic')

plt.figure(figsize=(12,8))

plt.subplot(121),plt.imshow(ASM_original_pic)
plt.title('ASM_original_Pic')

plt.subplot(122),plt.imshow(energy_original_pic)
plt.title('energy_original_pic')


# In[33]:


result1 = greycomatrix(picture_1, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[40]:


picture_1.shape


# In[55]:


plt.imshow(picture_1)


# In[34]:


contrast_picture_1 = greycoprops(result1, 'contrast')
dissimilarity_picture_1 = greycoprops(result1, 'dissimilarity')
homogeneity_picture_1 = greycoprops(result1, 'homogeneity')
ASM_picture_1 = greycoprops(result1, 'ASM')
energy_picture_1 = greycoprops(result1, 'energy')


# In[35]:


plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(picture_1)
plt.title('sample image 1')

plt.subplot(142),plt.imshow(contrast_picture_1)
plt.title('contrast_picture_1')

plt.subplot(143),plt.imshow(dissimilarity_picture_1)
plt.title('dissimilarity_picture_1')

plt.subplot(144),plt.imshow(homogeneity_picture_1)
plt.title('homogenity_picture_1')

plt.figure(figsize=(12,8))

plt.subplot(121),plt.imshow(ASM_picture_1)
plt.title('ASM_picture_1')

plt.subplot(122),plt.imshow(energy_picture_1)
plt.title('energy_picture_1')


# In[36]:


result2 = greycomatrix(picture_2, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[39]:


picture_2.shape


# In[56]:


plt.imshow(picture_2)


# In[37]:


contrast_picture_2 = greycoprops(result2, 'contrast')
dissimilarity_picture_2 = greycoprops(result2, 'dissimilarity')
homogeneity_picture_2 = greycoprops(result2, 'homogeneity')
ASM_picture_2 = greycoprops(result2, 'ASM')
energy_picture_2 = greycoprops(result2, 'energy')


# In[38]:


plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(picture_2)
plt.title('sample image 2')

plt.subplot(142),plt.imshow(contrast_picture_2)
plt.title('contrast_picture_2')

plt.subplot(143),plt.imshow(dissimilarity_picture_2)
plt.title('dissimilarity_picture_2')

plt.subplot(144),plt.imshow(homogeneity_picture_2)
plt.title('homogenity_picture_2')

plt.figure(figsize=(12,8))

plt.subplot(121),plt.imshow(ASM_picture_2)
plt.title('ASM_picture_2')

plt.subplot(122),plt.imshow(energy_picture_2)
plt.title('energy_picture_2')

plt.show()


# In[47]:


plt.figure(figsize=(15,8))

plt.subplot(131),plt.scatter(contrast_original_pic, dissimilarity_original_pic)
plt.title('contrast_original vs dissimilarity_original')

plt.subplot(132),plt.scatter(contrast_picture_1, dissimilarity_picture_1)
plt.title('contrast_picture_1 vs dissimilarity_picture_1')

plt.subplot(133),plt.scatter(contrast_picture_2, dissimilarity_picture_2)
plt.title('contrast_picture_2 vs dissimilarity_picture_2')

plt.show()

