import sys
import codecs

def preproc_1(filename, out_data_file, out_target_file, need_seg):
    fin = open(filename).readlines()
    fout_data = open(out_data_file, 'w')
    fout_target = open(out_target_file, 'w')
    if need_seg:
        fout_seg = open(filename + '_seg', 'w')
    for line in fin:
        if line.strip() == '':
            fout_data.write('\n')
            fout_target.write('\n')
            if need_seg:
                fout_seg.write('\n')
        else:
            feats = line.strip().decode('gbk').split()
            data = feats[:-1]
            target = feats[-1]
            for item in data:
                fout_data.write(item.encode('gbk') + ' ')
            fout_data.write('\n')
            fout_target.write(target.encode('gbk') + ' ')
            if need_seg:
                seg = feats[1]
                fout_seg.write(seg + ' ')

def preproc_2(filename, out_data_file, out_target_file, speaker, need_seg,
              encoding="gb18030"):
    """ add speaker tag in head and tail
    :param filename:
    :param out_data_file:
    :param out_target_file:
    :param need_seg:
    :return:
    """
    fin = codecs.open(filename, "r", encoding, "ignore").readlines()
    fout_data = codecs.open(out_data_file, 'w', encoding)
    fout_target = codecs.open(out_target_file, 'w', encoding)
    is_head = True
    if need_seg:
        fout_seg = codecs.open(filename + '_seg', 'w', encoding)
    for line in fin:
        if line.strip() == '':
            # add speaker tag in tail
            fout_data.write(speaker_tag + '\n')
            fout_target.write(speaker_lab + ' ')
            fout_data.write('\n')
            fout_target.write('\n')
            if need_seg:
                fout_seg.write('\n')
            is_head = True
        else:
            feats = line.strip().split()
            #feats = line.strip().decode('gbk').split()
            data = feats[:-1]
            target = feats[-1]
            if is_head:
                speaker_tag = " ".join(["<%s>" % speaker] + ["<UNK>"] * (len(data) - 1))
                speaker_lab = "T"
                fout_data.write(speaker_tag + '\n')
                fout_target.write(speaker_lab + ' ')
                is_head = False
            for item in data:
                fout_data.write(item.split("@")[0].strip() + ' ')
            fout_data.write('\n')
            fout_target.write(target.split("@")[0].strip() + ' ')
            if need_seg:
                seg = feats[1]
                fout_seg.write(seg + ' ')

    fout_data.close()
    fout_target.close()
    if need_seg:
        fout_seg.close()
if __name__ == '__main__':
    work_dir = r"D:/Python Scripts/baidu/personal-code-nzp/prosody_test/data"
    in_fn = "%s/miduo-A300.txt" % work_dir
    out_fea_fn = "%s/miduo-A300.fea" % work_dir
    out_tar_fn = "%s/miduo-A300.lab" % work_dir
    speakers = [None,   "f28",  "com",  "f20", "f7",
                "m15",  "yyjw", "gezi", "f11", "novel",
                "news", "miduo"]
    speaker = speakers[11]
    need_seg = 0
    preproc_2(in_fn, out_fea_fn, out_tar_fn, speaker, need_seg)
    #preproc_2(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
