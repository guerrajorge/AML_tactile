import rosbag
import yaml
import os
import message_converter

program_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])


def ros2xml(msg, name, depth=0):
    xml = ''
    tabs = '\t' * depth
    if hasattr(msg, '_type'):
        msg_type = msg._type
        xml = xml + tabs + '<' + name + ' type=\"' + msg_type + '\">\n'

        try:
            for slot in msg.__slots__:
                xml += ros2xml(getattr(msg, slot), slot, depth=depth+1)
        except:
            xml = xml + tabs + str(msg)
        xml = xml + tabs + '</' + name + '>\n'
    else:
        xml = xml + tabs + '<' + name + '>' + str(msg) + '</' + name + '>\n'
    return xml


def run():
    """
    Main run method. Calls other helper methods to get work done
    """
    # get list of only bag files in current dir.
    bag_file_list = [f for f in os.listdir(".") if f[-4:] == ".bag"]	

    number_of_files = str(len(bag_file_list))
    print 'reading all {0} bagfiles in current directory: \n'.format(number_of_files)
    for filename in bag_file_list:
        print filename
    print ''

    for filename in bag_file_list:
        bag = rosbag.Bag(filename)
        topic_list = read_bag_topic_list(bag)

    print 'topic in the current bag'
    for topic in topic_list:
        print topic
    print ''

    filename = filename.replace('.bag', '')
    folder_dir = os.path.join(program_path, filename)

    if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

    print 'Printing all topics'
    for topic in topic_list:
        extract_data(bag, topic, folder_dir)

        bag.close()


def extract_data(bag, topic, directory):
    """
    Spew messages to a file

    args:
        topic -> topic to extract and print to txt file
    """
    output_dir = os.path.join(directory, topic.replace('/', '')+'.yaml')
    print 'Printing {0}'.format(topic)
    print 'Output file will be called {0}.'.format(directory.split('/')[-1])

    output_file = open(output_dir, "w")

    for topic, msg, t in bag.read_messages(topics=topic):
        created_dictionary = message_converter.convert_ros_message_to_dictionary(msg)
        yaml.dump(created_dictionary, output_file,default_flow_style=False)

    output_file.close()
    print 'DONE'


def menu(topic_list):
    """
    Print the user menu and take input

    args:
        topicList: tuple containing list of topics

    returns:
        selection: user selection as integer
    """

    print 'Enter a topic number to extract raw data from:'
    selection = raw_input('>>>')
    if int(selection) == len(topic_list):
        return -92  # print all
    elif int(selection) == (len(topic_list) + 1):
        return -45  # exit
    elif (int(selection) < len(topic_list)) and (int(selection) >= 0):
        return int(selection)
    else:
        print '[ERROR] Invalid input'


def read_bag_topic_list(bag):
    """
    Read and save the initial topic list from bag
    """
    print 'Reading topics in {0} bag. Can take a while..'.format(bag.filename)
    topic_list = []
    for topic, msg, t in bag.read_messages():
        if topic_list.count(topic) == 0:
            topic_list.append(topic)

    print '{0} topics found:'.format(len(topic_list))
    return topic_list

if __name__ == '__main__':
    run()
