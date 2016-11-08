#!/usr/bin/env python
# encoding: utf-8

import json
import commands
import xml.etree.ElementTree

sample_list = "./all.list"

def checkVOC(annFile):
    root = xml.etree.ElementTree.parse(annFile).getroot();
    anns = root.findall('object')

    newAnns = []
    for ann in anns:
        name = ann.find('name').text
        newAnn = {}
        newAnn['category_id'] = name

        bbox = ann.find('bndbox')
        newAnn['bbox'] = [-1,-1,-1,-1]
        newAnn['bbox'][0] = float( bbox.find('xmin').text )
        newAnn['bbox'][1] = float( bbox.find('ymin').text )
        newAnn['bbox'][2] = float( bbox.find('xmax').text ) - newAnn['bbox'][0]
        newAnn['bbox'][3] = float( bbox.find('ymax').text ) - newAnn['bbox'][1]
        newAnn['iscrowd'] = int( ann.find('difficult').text )
        newAnns.append(newAnn)

    if len(newAnns) > 0:
        jobj= {}
        jobj["annotation"] = newAnns
        jobj["image"] = {}
        jobj["image"]["width"] = int(root.find("size").find("width").text)
        jobj["image"]["height"] = int(root.find("size").find("height").text)
        return jobj

    return None

with open(sample_list) as f:
    seq = 1
    for line in f:
        sample_info = line.split()

        bboxes = None

        if ( sample_info[2] == 'voc' ):
            bboxes = checkVOC(sample_info[1])
            bboxes["image"]["file"] = sample_info[0]

        if ( bboxes != None) :
            '''
            info_file = './infos/' + str(seq) + '_info.json'
            with open(info_file, "w") as fd:
                fd.write( json.dumps(bboxes) )
                print sample_info[0], info_file
                seq = seq + 1
            '''
            print json.dumps(bboxes)


