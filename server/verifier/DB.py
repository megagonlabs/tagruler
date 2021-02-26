from datetime import datetime
import json
import os
import pandas as pd
from verifier.translator import make_lf


class InteractionDBSingleton:
    filename = "interactionDB.json"
    def __init__(self):
        self.db = {}
        self.count = 0

    def add(self, interaction: dict):
        if "index" in interaction:
            index = interaction["index"]
            interaction['time_submitted'] = datetime.now()
            self.db[index].update(interaction)
        else:
            index = self.count
            interaction["index"] = index
            interaction['time_first_seen'] = datetime.now()
            self.db[index] = interaction
            self.count += 1
        return index

    def update(self, index: int, selected_lf_ids: list):
        self.db[index]["lfs"] = selected_lf_ids

    def get(self, index: int):
        return self.db[index]

    def save(self, dirname):
        with open(os.path.join(dirname, self.filename), "w+") as file:
            json.dump({
                "db": self.db,
                "count": self.count
                }, file, default=str)

    def load(self, dirname):
        with open(os.path.join(dirname, self.filename), "r") as file:
            data = json.load(file)
            self.db = data['db']
            self.count = data['count']

interactionDB = InteractionDBSingleton()

class LFDBSingleton:
    filename = "LF_DB.json"
    def __init__(self):
        self.db = {}
        self.lf_index = 0

    def get(self, lf_id):
        return self.db[lf_id]

    def add_lfs(self, lf_dicts: dict, all_concepts, emb_dict, ui_label_dict):
        new_lfs = {}
        for lf_hash, lf_explanation in lf_dicts.items():
            if not lf_hash in self.db:
                lf_explanation["time_submitted"] = datetime.now()
                lf_explanation["ID"] = self.lf_index
                lf_explanation["active"] = True
                self.lf_index += 1

                crnt_lf = make_lf(lf_explanation, all_concepts.get_dict(), emb_dict, ui_label_dict)
                self.db[lf_hash] = lf_explanation
                new_lfs[lf_hash] = crnt_lf
            else:
                self.db[lf_hash]["active"] = True
                new_lfs[lf_hash] = make_lf(self.db[lf_hash], all_concepts.get_dict(), emb_dict, ui_label_dict)
            
        return new_lfs

    def delete(self, lf_id: str):
        return self.db.pop(lf_id)

    def deactivate(self, lf_id: str):
        self.db[lf_id]["active"] = False
        return self.db[lf_id]

    def update(self, stats: dict):
        for lf_id, stats_dict in stats.items():
            self.db[lf_id].update(stats_dict)
        return self.db.copy()

    def __contains__(self, item: str):
        return item in self.db

    def __len__(self):
        return len(self.db)

    def save(self, dirname):
        with open(os.path.join(dirname, self.filename), "w+") as file:
            json.dump({
                "db": self.db,
                "lf_index": self.lf_index
                }, file, default=str)

    def load(self, dirname, all_concepts):
        with open(os.path.join(dirname, self.filename), "r") as file:
            data = json.load(file)
            lfs = data['db']
            self.db.update({k:v for k,v in lfs.items() if not v['active']})
            self.lf_index = data['lf_index']
            return self.add_lfs({k:v for k,v in lfs.items() if v['active']}, all_concepts)


LF_DB = LFDBSingleton()