from black import out
import tensorflow as tf
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def parse_single_image(image, label):
    #define the dictionary -- the structure -- of our single example
    data = {
            'height' : _int64_feature(image.shape[0]),
            'width' : _int64_feature(image.shape[1]),
            'channels' : _int64_feature(label.shape[2]),
            'raw_image' : _bytes_feature(serialize_array(image)),
        }
    #print(label.shape[2])
    for elem in range(label.shape[2]):
            data['label'+str(elem)] = _bytes_feature(serialize_array(label[:,:,elem]))

    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def write_images_to_tfr_short(images, labels, folder, filename:str="images"):
  filename = filename+".tfrecords"
  filename = folder + filename
  writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
  count = 0

  for index in range(len(images)):
    #get the data we want to write
    current_image = images[index] 
    current_label = labels[index]
    out = parse_single_image(image=current_image, label=current_label)
    writer.write(out.SerializeToString())
    count += 1
  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count