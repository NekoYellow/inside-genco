"""Scientific boundaries: sharded data, stable identities, exposure and selections."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import baseline
from research_data import (network_identity, assign_groups, load_selection, file_hash,
                           BUS_COLUMNS,GEN_COLUMNS,BRANCH_COLUMNS)


def fixtures():
    b=pd.DataFrame(0.,index=range(3),columns=BUS_COLUMNS)
    b['bus']=[0,1,2];b['PQ']=[0,1,1];b['REF']=[1,0,0];b['vn_kv']=138.
    g=pd.DataFrame([[4,0,1,1],[5,1,0,0]],columns=GEN_COLUMNS)
    e=pd.DataFrame(0.,index=range(2),columns=BRANCH_COLUMNS)
    e['idx']=[10,11];e['from_bus']=[0,1];e['to_bus']=[1,2];e['br_status']=1;e['r']=.01;e['x']=.1;e['tap']=1.
    return b,g,e


class ResearchTests(unittest.TestCase):
    def test_row_order_and_targets_do_not_change_identities(self):
        b,g,e=fixtures();a=network_identity(b,g,e)
        b['Vm']=np.nan;b['Qg']=999.;g['p_mw']=-999.;e['pf']=np.nan
        z=network_identity(b.iloc[::-1],g.iloc[::-1],e.iloc[::-1])
        self.assertEqual(a,z)

    def test_electrical_and_topological_variants_are_distinct(self):
        b,g,e=fixtures();a=network_identity(b,g,e)
        e.loc[0,'Yff_r']+=.01;z=network_identity(b,g,e)
        self.assertEqual(a['topology_id'],z['topology_id']);self.assertNotEqual(a['electrical_id'],z['electrical_id'])
        e.loc[0,'br_status']=0;z=network_identity(b,g,e)
        self.assertNotEqual(a['topology_id'],z['topology_id'])
        b.loc[2,['PQ','PV']]=[0,1];w=network_identity(b,g,e)
        self.assertNotEqual(z['node_map_id'],w['node_map_id'])

    def test_invalid_device_identity_rejected(self):
        b,g,e=fixtures();e.loc[1,'idx']=10
        with self.assertRaisesRegex(ValueError,'Duplicate'):network_identity(b,g,e)

    def test_exposure_and_group_isolation(self):
        groups=assign_groups(range(996),range(46))
        self.assertEqual([len(v) for v in groups.values()],[400,400,196])
        self.assertTrue(set(range(46))<=set(groups['exploration']))
        self.assertEqual(set().union(*map(set,groups.values())),set(range(996)))
        self.assertFalse(set(groups['exploration'])&set(groups['confirmation']))
        self.assertEqual(groups,assign_groups(reversed(range(996)),range(46)))

    def test_multifile_partition_is_read_once_and_all_shards_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[]
            for table in ('bus','gen','branch'):
                for suffix,ids in [('a',[10,11]),('b',[12,13])]:
                    rel=f'{table}_data.parquet/scenario_partition=7/{suffix}.parquet'
                    path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                    pd.DataFrame({'scenario':ids}).to_parquet(path,index=False)
                    entries.append(dict(path=rel,local=rel))
            with patch.object(baseline,'ROOT',root):
                plan,required=baseline.plan_partitions([10,13],entries)
            self.assertEqual(plan,[('scenario_partition=7',[10,13])]);self.assertEqual(len(required),6)

    def test_selection_rejects_duplicates_train_ids_and_mutated_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'ids.json'
            for ids in ([1,1],[0,1]):
                path.write_text(json.dumps(ids))
                with self.assertRaises(ValueError):load_selection(path,[1,2,3],root)
            index=root/'index.csv'
            pd.DataFrame([dict(scenario_id=1,research_split='exploration',topology_id='a')]).to_csv(index,index=False)
            path.write_text(json.dumps(dict(scenario_ids=[1],name='pilot',index_path='index.csv',
                research_split='exploration',topology_id='a',dependencies={'index.csv':file_hash(index)})))
            self.assertEqual(load_selection(path,[1,2,3],root)[0],[1])
            index.write_text(index.read_text().replace('exploration','confirmation'))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_selection(path,[1,2,3],root)


if __name__=='__main__':unittest.main()
