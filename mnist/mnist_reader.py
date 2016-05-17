import os
import struct
from array import array

class MNIST(object):
  def __init__(self, path='.'):
    self.path = path

    self.test_img_fname = 'raw_data/t10k-images-idx3-ubyte'
    self.test_lbl_fname = 'raw_data/t10k-labels-idx1-ubyte'

    self.train_img_fname = 'raw_data/train-images-idx3-ubyte'
    self.train_lbl_fname = 'raw_data/train-labels-idx1-ubyte'

    self.test_images = []
    self.test_labels = []

    self.train_images = []
    self.train_labels = []

  def load_testing(self):
    ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                     os.path.join(self.path, self.test_lbl_fname))

    self.test_images = ims
    self.test_labels = labels

    return ims, labels

  def load_training(self):
    ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                     os.path.join(self.path, self.train_lbl_fname))

    self.train_images = ims
    self.train_labels = labels

    return ims, labels

  @classmethod
  def load(cls, path_img, path_lbl):
    with open(path_lbl, 'rb') as file:
      magic, size = struct.unpack(">II", file.read(8))
      if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049,'
          'got %d' % magic)

      labels = array("B", file.read())

    with open(path_img, 'rb') as file:
      magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
      if magic != 2051:
        raise ValueError('Magic number mismatch, expected 2051,'
          'got %d' % magic)

      image_data = array("B", file.read())

    images = []
    for i in xrange(size):
      images.append([0]*rows*cols)

    for i in xrange(size):
      images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

    return images, labels

  def write_files(self):
    test_img, test_label = self.load_testing()
    train_img, train_label = self.load_training()
    assert len(test_img) == len(test_label)
    assert len(test_img) == 10000
    assert len(train_img) == len(train_label)
    assert len(train_img) == 60000

    f = open("data/test_images.txt", "w")
    for i in xrange(len(test_img)):
      f.write(' '.join(str(k) for k in test_img[i]) + '\n')
    f.close()

    f = open("data/test_labels.txt", "w")
    for i in xrange(len(test_label)):
      f.write(str(test_label[i]) + '\n')
    f.close()

    f = open("data/train_images.txt", "w")
    for i in xrange(len(train_img)):
      f.write(' '.join(str(k) for k in train_img[i]) + '\n')
    f.close()

    f = open("data/train_labels.txt", "w")
    for i in xrange(len(train_label)):
      f.write(str(train_label[i]) + '\n')
    f.close()

#    for i in xrange(len(test_img)):
#      # write each image in its own line
#      print test_label[i],
#      for j in xrange(len(test_img[i])):
#        print test_img[i][j],
#      print

#    print 'Showing num:%d' % train_label[5]
#    print self.display(train_img[5])
#    print
    return True

  @classmethod
  def display(cls, img, width=28, threshold=200):
    # img is a list of pixel values
    print img

#    render = ''
#    for i in range(len(img)):
#      if i % width == 0: render += '\n'
#      if img[i] > threshold:
#        render += '@'
#      else:
#        render += '.'
#    return render


if __name__ == "__main__":
  mn = MNIST('.')
  mn.write_files()

