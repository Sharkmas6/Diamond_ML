import sqlite3


class AiLabelParser(object):
    '''
    A class to parse MR related details
    '''

    def __init__(self, handle):
        '''
        Initialise the class with the handle
        '''
        self.handle = sqlite3.connect(handle)
        self.cur = self.handle.cursor()
        self.found_ep_data = []
        self.found_no_ep_data = []

    def _add_success_column(self, col_name="IS_SUCCESS"):
        '''
        Add column to EP_STATS table
        '''

        # check if column already exists
        self.cur.execute(rf"PRAGMA table_info(EP_STATS)")
        cols = [info[1] for info in self.cur.fetchall()]

        if col_name not in cols:
            self.cur.execute(f'''
          ALTER TABLE EP_STATS ADD {col_name} int
          ''')
            self.cur.fetchall()

    def _parse_ai_labels(self, pdb_id):
        '''
        Get AI labels for EP (for PDB id and sweep_id) and MR success (for PDB id)
        '''
        print('Getting AI labels for PDB %s' % pdb_id)

        # get dataset ids for each pdb code
        self.cur.execute('''
      SELECT id FROM DATASET_INFO WHERE PDB_CODE="%s"
      ''' % pdb_id)
        dataset_ids = [i[0] for i in self.cur.fetchall()]

        # get cc_original, cc_inverse and acl
        self.cur.execute(f'''
      SELECT SHELXE_CC_ORIGINAL, SHELXE_CC_INVERSE, SHELXE_ACL_ORIGINAL, SHELXE_ACL_INVERSE
      FROM EP_STATS WHERE DATASET_id IN ({','.join(['?']*len(dataset_ids))})
      ''', dataset_ids)
        ccs_acl = self.cur.fetchall()

        # check if algorithms were successful
        success = []
        for cc_original, cc_inverse, acl_original, acl_inverse in ccs_acl:
            success += [(abs(cc_original - cc_inverse) > 10) and
                        ((cc_original > 25 and acl_original > 10) ^ (cc_inverse > 25 and acl_inverse > 10))]

        if success == []:
            print('No EP datum found')

            return False

        else:
            # update table
            for suc, ds_id in zip(success, dataset_ids):
                self.cur.execute(f'''
              UPDATE EP_STATS SET IS_SUCCESS={int(suc)} WHERE DATASET_id={ds_id}
              ''')
                self.cur.fetchall()
            print("EP results successfully parsed")

            return True

    def add_entry(self, pdb_id):
        '''
        Add protein details to database
        '''
        self._add_success_column()
        found_data = self._parse_ai_labels(pdb_id)

        self.handle.commit()
        return found_data

    def add_all_entries(self):
        '''
        Add all proteins' details to database
        '''
        self.cur.execute('''
        SELECT DISTINCT PDB_CODE FROM PDB_DATA
      ''')

        for pdb_code in self.cur.fetchall():
            if self.add_entry(pdb_id=pdb_code[0]):
                self.found_ep_data += [pdb_code[0]]
            else:
                self.found_no_ep_data += [pdb_code[0]]

        print(f"Found EP datum for {len(self.found_ep_data)} entities\n"
              f"No EP datum found for {len(self.found_no_ep_data)} entities")


database_path = r"path/to/database"
parser = AiLabelParser(database_path)
parser.add_all_entries()
