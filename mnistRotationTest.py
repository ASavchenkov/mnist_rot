import tensorflow as tf
import glob
import numpy as np
import time
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
def gen_file_names_and_labels(rootDir):
    
    """goes through the directory structure and extracts images and labels from each image."""
    file_names = []
    labels = []
    rotations = []
    for file_name in glob.glob(rootDir+'/*/*'):
         
        file_type_removed = file_name.split('.')[0]
        split_by_dir = file_type_removed.split('/')

        file_names.append(file_name)

        labels.append(int(split_by_dir[2])) #getting the folder it's in, turning into an int, and using as label
        
        rotations.append(int(split_by_dir[3].split('_')[1]))

    return file_names, labels, rotations

def read_images_from_disk(input_queue):
    contents = tf.read_file(input_queue[0])
    label = tf.one_hot(input_queue[1],depth=10,dtype=tf.float32)
    rotation = input_queue[2]
    
    image = tf.image.decode_png(contents,channels=1)
    castedImage = tf.cast(image,tf.float32)
    scaledImage = tf.div(castedImage,256)

    return scaledImage,label,rotation

def inputPipeline(rootDir,batch_size):
    #just make lists of all the files. regular python function
    image_list, label_list, rot_list = gen_file_names_and_labels(rootDir)

    images = tf.convert_to_tensor(image_list,dtype=tf.string)
    labels = tf.convert_to_tensor(label_list,dtype=tf.int32)
    rotations = tf.convert_to_tensor(rot_list)
    rotations_casted = tf.cast(rotations, tf.float32)
    rotations_scaled = rotations_casted/360
    
    input_queue = tf.train.slice_input_producer([images,labels,rotations_scaled], shuffle=True)
    image, label, rotation = read_images_from_disk(input_queue)
    image_reshaped= tf.reshape(image,[28,28,1])

    return tf.train.batch([image_reshaped,label,rotation],batch_size = batch_size)
    

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    
    #THERE A PIPELINE FOR BOTH TESTING AND TRAINING. THEY COME IN PAIRS    
    image_batch, label_batch, rotation_batch = inputPipeline('mnist_png_rot/training',50)
    image_batch_t, label_batch_t, rotation_batch_t = inputPipeline('mnist_png_rot/testing',100)
    
    #finally, these down here are the "inputs" to the architecture
    #there should only be one set of these at this stage in the code.
    x = tf.placeholder_with_default(image_batch,shape = [None,28,28,1])
    y_ = tf.placeholder_with_default(label_batch,shape = [None,10])
    y_r= tf.placeholder_with_default(rotation_batch,shape = [None])



    #convolution 1
    W_conv1 = weight_variable([5,5,1,32])                   #define weights
    b_conv1 = bias_variable([32])                           #define biases
    h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)       #do convolution
    h_pool1 = max_pool_2x2(h_conv1)                         #do pooling

    #convolution 2
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #fully connected layer 1
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])          #flatten pooled layer
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

    #dropout layer
    keep_prob = tf.placeholder(tf.float32)#probability that stuff will get dropped
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
   
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    
    W_fc2_r = weight_variable([1024,1])
    b_fc2_r = bias_variable([1])

    y_conv      = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
    y_conv_r    = tf.sigmoid(tf.matmul(h_fc1_drop,W_fc2_r)+ b_fc2_r)

    #part that measures correctness
    cross_entropy   = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
    rotation_mean_squared  = tf.reduce_mean(tf.square(y_conv_r-y_r))
    class_weight = 1.0;
    rot_weight = 0.0; #just doing regular classification. not training rotation
    total_loss = ((cross_entropy*class_weight) + (rotation_mean_squared*rot_weight))
    #weighted average of losses

    #part that trains
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
    
    #parts that outputs accuracy on classifying a batch
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    rotation_error = rotation_mean_squared  #this is already a good measure of accuracy,
                                            #since we're dealing with a continuous function,
                                            #not one hot encoded classifications.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.initialize_all_variables())
    print(y_conv_r.eval(feed_dict={keep_prob:1.0}))
    # time.sleep(10) 
    for i in range(500):
     
        if i%20 ==0:
            x_test, y_test, y_r_test = sess.run([image_batch_t,label_batch_t,rotation_batch_t])
            train_accuracy, rot_err, raw_output = sess.run([accuracy,rotation_error,y_conv_r],feed_dict={keep_prob:1.0,x:x_test,y_:y_test,y_r:y_r_test})
            print(y_r_test,raw_output)
            print("step %d, training accuracy %g rot_err: %g"%(i, train_accuracy,rot_err))
        # if i%1000==0:
            # accuracies = []
            # for i in range(100):
                # x_test, y_test, y_r_test = sess.run([image_batch_t,label_batch_t,rotation_batch_t])
                # accuracies.append(accuracy.eval(feed_dict={keep_prob:1.0,x:x_test,y_:y_test,y_r:y_r_test}))
                # print("accuracy after {} iterations is: {}".format(i,np.mean(accuracies)))
        train_step.run(feed_dict={keep_prob:0.9})


