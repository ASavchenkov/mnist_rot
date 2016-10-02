import tensorflow as tf
import glob
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
    for file_name in glob.glob(rootDir+'/*/*'):
         
        file_type_removed = file_name.split('.')[0]
        split_by_dir = file_type_removed.split('/')
        file_names.append(file_name)
        labels.append(int(split_by_dir[2])) #getting the folder it's in, turning into an int, and using as label
    return file_names, labels

def read_images_from_disk(input_queue):
    label = tf.one_hot(input_queue[1],depth=10,dtype=tf.float32)
    contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(contents,channels=1)
    castedImage = tf.cast(image,tf.float32)
    scaledImage = tf.div(castedImage,256)
    # scaledImage = castedImage
    return scaledImage,label

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    
    #THERE A PIPELINE FOR BOTH TESTING AND TRAINING. THEY COME IN PAIRS    
    image_list_train,   label_list_train    = gen_file_names_and_labels('mnist_png/training')
    image_list_test,    label_list_test     = gen_file_names_and_labels('mnist_png/testing')

    images_train    = tf.convert_to_tensor(image_list_train,dtype=tf.string)    
    images_test     = tf.convert_to_tensor(image_list_test,dtype=tf.string)    

    #remember that these aren't the actual images, just file_names
    labels_train    = tf.convert_to_tensor(label_list_train,dtype=tf.int32)
    labels_test     = tf.convert_to_tensor(label_list_test,dtype=tf.int32)

    input_queue_train   = tf.train.slice_input_producer([images_train   ,labels_train]  , shuffle=True)
    input_queue_test    = tf.train.slice_input_producer([images_train   ,labels_train]  , shuffle=True)
    
    #now we need to make tensorflow switch between train and test based on a variable we pass it 
    asdf = tf.placeholder(tf.int32)
    input_queue = tf.cond( asdf>0, lambda: input_queue_train, lambda: input_queue_test)
    # input_queue = input_queue_test
    image, label = read_images_from_disk(input_queue)
    image_reshaped = tf.reshape( image, [28,28,1])
    image_batch, label_batch = tf.train.batch([image_reshaped,label],batch_size=50)
    #COMMENTED OUT CODE INVOLVES FEEDING VALUES USING THEIR OWN FEED_DICT THING   
    # x = tf.placeholder(tf.float32, shape=[28,28,1])
    # y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # x = tf.reshape(image_batch,[-1,28,28,1])
    x = image_batch
    y_ = label_batch
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
    keep_prob = tf.placeholder(tf.float32)#probability that stuff will get dropped?
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
   
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))#part that measures correctness
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #part that trains
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1)) #parts that output actual accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.initialize_all_variables())
    print(label_batch.eval(feed_dict={asdf:0,keep_prob:1.0}))
    for i in range(500):
        # batch = mnist.train.next_batch(50)
     
        if i%20 ==0:
            train_accuracy = accuracy.eval(feed_dict={keep_prob:1.0,asdf:0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={keep_prob:0.9,asdf:0})


