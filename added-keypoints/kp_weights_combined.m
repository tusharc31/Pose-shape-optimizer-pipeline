function W = kp_weights_combined(seq, frm, id)

[w, wkps, wkpl] = kp_weights(seq, frm, id);
W = 0.8 * wkps + 0.2 * wkpl;

end