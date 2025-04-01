#!/usr/bin/env python
# coding: utf-8

# # Load Raw Data

# In[1]:


import os
import numpy as np
import mne
from mne.datasets import sample
import matplotlib.pyplot as plt
from mne.inverse_sparse import gamma_map, make_stc_from_dipoles
from mne.viz import (
    plot_dipole_amplitudes,
    plot_dipole_locations,
    plot_sparse_source_estimates,
)


data_path = mne.datasets.sample.data_path()

sample_data_raw_file = os.path.join(
    data_path, "MEG", "sample", "sample_audvis_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)


# # Pre-processing

# In[2]:


raw.filter(1.0, 40.0, fir_design='firwin', skip_by_annotation='edge')
#
ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=1)
ica.fit(raw)
#
ica.plot_sources(raw, show_scrollbars=False)
#


# In[3]:


#remove artifacts
ica.exclude = [0,1]
ica.apply(raw)
#ica.plot_sources(raw, show_scrollbars=False)  #To see the artifact is removed


# # Events and Evoked 

# In[4]:


events = mne.find_events(raw)
event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "smiley": 5,
    "buttonpress": 32,
}
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.7, event_id=event_dict, preload=True)
#
a_r = epochs["auditory/right"].average()
a_l = epochs["auditory/left"].average()
#
#fig = a_l.plot_joint(picks='mag')
fig = a_l.plot_joint([ 0.075, 0.100], picks='mag',title='Left Auditory')


# # visual evoked response

# In[5]:


v_r = epochs["visual/right"].average()
v_l = epochs["visual/left"].average()
fig = v_r.plot_joint(picks='mag', title='Right Visual')


# # Source localization

# In[6]:


subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
evoked_fname = meg_path / "sample_audvis-ave.fif"
cov_fname = meg_path / "sample_audvis-cov.fif"
# Read the forward solution
forward = mne.read_forward_solution(fwd_fname)

# Read noise noise covariance matrix and regularize it
cov = mne.read_cov(cov_fname)
cov = mne.cov.regularize(cov, a_l.info, rank=None)

#Call Gamma Map function
a_l.set_eeg_reference(projection=True)
alpha = 0.2
dipoles, residual = gamma_map(
    a_l,
    forward,
    cov,
    alpha,
    xyz_same_gamma=True,
    return_residual=True,
    return_as_dipoles=True,
)


# # Plot Dipoles

# In[7]:


#plot_dipole_amplitudes(dipoles)
 # Plot dipole locations of all dipoles with MRI slices
print(f"\n Dipoles length=",len(dipoles))
for dip in dipoles:
    plot_dipole_locations(dip, forward['mri_head_t'], 'sample',subjects_dir=subjects_dir, mode='orthoview', idx='amplitude')


# # Plot Brain 3D

# In[8]:


stc = make_stc_from_dipoles(dipoles, forward["src"])
scale_factors = np.max(np.abs(stc.data), axis=1)
scale_factors = 0.5 * (1 + scale_factors / np.max(scale_factors))

plot_sparse_source_estimates(
    forward["src"],
    stc,
    bgcolor=(1, 1, 1),
    modes=["cone"],
    opacity=0.2,
    scale_factors=(scale_factors, None),
    fig_name="Gamma-MAP",
)


# In[9]:


#plot Brain with time line
p6 = stc.plot(
    hemi="both",
    subjects_dir=subjects_dir,
    smoothing_steps=5,
    clim = 'auto',
    time_unit= 's',
    title = f'alpha = {alpha}',
    )



# # Source localization for visual stimilus

# In[14]:


#Call Gamma Map function
v_r.set_eeg_reference(projection=True)
alpha = 0.2
v_dipoles, v_residual = gamma_map(
    v_r,
    forward,
    cov,
    alpha,
    xyz_same_gamma=True,
    return_residual=True,
    return_as_dipoles=True,
)


# In[15]:


#plot
stc = make_stc_from_dipoles(v_dipoles, forward["src"])
p6 = stc.plot(
    hemi="both",
    subjects_dir=subjects_dir,
    smoothing_steps=5,
    clim = 'auto',
    time_unit= 's',
    title = f'alpha = {alpha}',
    )


# In[ ]:




