import numpy as np


class Storage:

    def __init__(self, start, domain, method, options):
        self.this_temp_index = 0
        self.max_temp_index = method.storage_size
        self.this_perm_index = 0

        self.temp_storage = method.init_temp_storage(start, domain, options)

        self.perm_storage = {}
        for req in options.repo.perm_requests:
            if req in method.temp_storage:
                perm_item = method.temp_storage[req][-1]
            else:
                # perm_item = req(start)
                perm_item = start
            self.perm_storage[req] = [perm_item]

    def book_keeping(self, temp_storage):

        # Retrieve New Data
        # new_data = temp_storage['Data'][-1]
        new_data = temp_storage['Policy'][-1]

        # update temp_storage
        self.temp_storage = temp_storage

        # update PermStorage
        self.this_perm_index += 1
        for req in self.perm_storage:
            if req in self.temp_storage:
                perm_item = self.temp_storage[req][-1]
            else:
                # print req
                perm_item = req(new_data)
            self.perm_storage[req].append(perm_item)
