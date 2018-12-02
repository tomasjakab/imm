import importlib



def import_dataset(dataset_name):
  dataset_filename = "imm.datasets." + dataset_name + "_dataset"
  datasetlib = importlib.import_module(dataset_filename)
  dset_class = None
  target_dataset_name = dataset_name.replace('_', '') + 'dataset'
  for name, cls in datasetlib.__dict__.items():
          if name.lower() == target_dataset_name.lower():
            dset_class = cls
  return dset_class
