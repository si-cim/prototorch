"""Tecator dataset for classification

URL:
    http://lib.stat.cmu.edu/datasets/tecator

LICENCE / TERMS / COPYRIGHT:
    This is the Tecator data set: The task is to predict the fat content
    of a meat sample on the basis of its near infrared absorbance spectrum.
    -------------------------------------------------------------------------
    1. Statement of permission from Tecator (the original data source)

    These data are recorded on a Tecator Infratec Food and Feed Analyzer
    working in the wavelength range 850 - 1050 nm by the Near Infrared
    Transmission (NIT) principle. Each sample contains finely chopped pure
    meat with different moisture, fat and protein contents.

    If results from these data are used in a publication we want you to
    mention the instrument and company name (Tecator) in the publication.
    In addition, please send a preprint of your article to

        Karin Thente, Tecator AB,
        Box 70, S-263 21 Hoganas, Sweden

    The data are available in the public domain with no responsability from
    the original data source. The data can be redistributed as long as this
    permission note is attached.

    For more information about the instrument - call Perstorp Analytical's
    representative in your area.

Description:
    For each meat sample the data consists of a 100 channel spectrum of
    absorbances and the contents of moisture (water), fat and protein.
    The absorbance is -log10 of the transmittance
    measured by the spectrometer. The three contents, measured in percent,
    are determined by analytic chemistry.
"""

import os

import numpy as np
import torch
from torchvision.datasets.utils import download_file_from_google_drive

from prototorch.datasets.abstract import ProtoDataset


class Tecator(ProtoDataset):
    """Tecator dataset for classification"""
    resources = [('1MMuUK8V41IgNpnPDbg3E-QAL6wlErTk0',
                  'ba5607c580d0f91bb27dc29d13c2f8df')]
    classes = ['0 - low_fat', '1 - high_fat']

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def download(self):
        """Download the data if it doesn't exist in already."""
        if self._check_exists():
            return

        if self.verbose:
            print('Making directories...')
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        if self.verbose:
            print('Downloading...')
        for fileid, md5 in self.resources:
            filename = 'tecator.npz'
            download_file_from_google_drive(fileid,
                                            root=self.raw_folder,
                                            filename=filename,
                                            md5=md5)

        if self.verbose:
            print('Processing...')
        with np.load(os.path.join(self.raw_folder, 'tecator.npz'),
                     allow_pickle=False) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        training_set = [torch.as_tensor(x_train), torch.as_tensor(y_train)]
        test_set = [torch.as_tensor(x_test), torch.as_tensor(y_test)]

        with open(os.path.join(self.processed_folder, self.training_file),
                  'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file),
                  'wb') as f:
            torch.save(test_set, f)

        if self.verbose:
            print('Done!')
