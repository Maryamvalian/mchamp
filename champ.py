#st
# get the data
    sel = [evoked.ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]
