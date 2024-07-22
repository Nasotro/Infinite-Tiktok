import os
import requests
import numpy as np
from moviepy.editor import *
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from google_images_search import GoogleImagesSearch
import shutil
import nltk
from PIL import Image

# configuring imageMagick
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})


def onlychars(text):
    return ''.join(c if c.isalnum() or c.isspace() else '' for c in text)

api_key = os.environ.get('OPENAI_API_KEY_LORRAIN')
client = OpenAI(api_key=api_key)
def generate_voice(text, output_filename, model="tts-1", voice="alloy"):
    """
    Converts a text string to speech using the OpenAI API and writes the resulting audio to an MP3 file.

    Args:
    text (str): The text string to convert to speech.
    output_filename (str): The name of the output MP3 file.
    model (str): The model to use for speech synthesis. Defaults to 'tts-1'.
    voice (str): The voice to use for speech synthesis. Defaults to 'alloy'.
    """
    global client
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY_LORRAIN')) if client is None else client
    speech_file_path = output_filename


    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(speech_file_path+('.mp3' if not speech_file_path.endswith('.mp3') else ''))
def getTextTimingsOfMp3(mp3file):
    global client
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY_LORRAIN')) if client is None else client
    
    audio_file = open(mp3file, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

    return transcript


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_description_pictures(story):
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "open-mistral-7b"

    client = MistralClient(api_key=api_key)

    preprompt = """Your job will be to write the visual of a tiktok video: the images that show on screen. You have to find some nice moments in the story to place pictures that will show on screen. Tell me when you choose an image of what. In this format:
    DURING THE SENTENCE "[place the full sentence here]", SHOW AN IMAGE OF "[place the description of the image]".
    Answer only with the descriptions of all the images. 
    Try to understand the story and which words are important to show visually. for example if you want to show an album cover, try to give the name of the album and the artist. If you want to show a person, don't try to give the name of the person, but just the description of what they are doing because is very unlikely I'll be able to find this exact person doing this exact action.
    Please chose the most images as possible (dont be shy ;)) and if a sentence needs more than one image, you can split the sentence in two or more parts.
    But NEVER give two descriptions for the same sentence.  
    Try not to give a description, but key words because i will look online for the exact same words you describe. That means if you are too complex, i will not be able to find the image.
    Don't forget that the context will not be given to the website where I will find the images, that means if the description is too vague, I will not be able to find the image.
    Try to kep the descriptions short and concise.
    Before I give you an example, there is one last important thing : You need to choose a lot of moments, because the images will be shown for a short time, so you need to have a lot of images to show the whole story.
    
    ----------------
    For example:
    
    STORY : "Eonis, a wise and patient god, was intrigued by a Rubik's Cube he saw a human child playing with. He materialized one and began to solve it, but found it surprisingly challenging. Despite his divine knowledge, it took him months in the celestial realm to finally align the last color."
    
    DURING THE SENTENCE "Eonis, a wise and patient god", SHOW AN IMAGE OF "a god".
    DURING THE SENTENCE "was intrigued by a Rubik's Cube he saw a human child playing with", SHOW AN IMAGE OF "a rubik's cube".
    DURING THE SENTENCE "He materialized one and began to solve it", SHOW AN IMAGE OF "a person solving a rubik's cube".
    DURING THE SENTENCE "but found it surprisingly challenging", SHOW AN IMAGE OF "someone thinking very hard".
    DURING THE SENTENCE "Despite his divine knowledge, it took him months", SHOW AN IMAGE OF "a hourglass".
    DURING THE SENTENCE "in the celestial realm to finally align the last color", SHOW AN IMAGE OF "a solved rubiks cube".
    ----------------
    
    Now it is your turn:

    STORY : """

    messages = [
        ChatMessage(role="user", content=preprompt + f'"{story}"'),
    ]

    chat_response = client.chat(
        model=model,
        messages=messages,
    )

    results = chat_response.choices[0].message.content

    return results.split("\n")
def convert_to_dict(lines):
    visual_dict = {}
    for line in lines:
        if(', SHOW AN IMAGE OF ' not in line):
            continue
        print(line)
        if('"' not in line[len(line)-5:]):
            line = line + '"'
        if line:
            sentence, image_desc = line.split(', SHOW AN IMAGE OF ', 1)
            image_desc = image_desc.replace('"', '')
            sentence = sentence.split(' THE SENTENCE ')[1]
            sentence = str(sentence).replace('"', '', 2)
            print("\t sentence : ", sentence)
            print("\t image_desc : ", image_desc)
            visual_dict[sentence] = image_desc
    return visual_dict
def get_pictures(story):
    return convert_to_dict(get_description_pictures(story))

def download_images(search_names, dirpath, max = -1):
    urls = []
    for i,search_term in enumerate(search_names):
        if(max != -1 and i >= max):
            break
        gis = GoogleImagesSearch(os.environ['GOOGLE_API_KEY'] , os.environ['GOOGLE_CX_KEY'])
        _search_params = {
            'q': search_term,
        }
        gis.search(search_params=_search_params, path_to_dir=dirpath, custom_image_name=str(i).zfill(3))
        for result in gis.results():
            urls.append(result.url)
    return urls
def convert_folder_to_png(input_folder):
    for filename in os.listdir(input_folder):
        print(filename)
        if(filename.endswith(".png")):
            continue
        try:
            img = Image.open(os.path.join(input_folder, filename))
            img = img.convert("RGBA")
            output_path = os.path.join(input_folder, os.path.splitext(filename)[0] + ".png")
            img.save(output_path, "PNG")

            os.remove(os.path.join(input_folder, filename))
        except IOError:
            # Not an image file, skip it
            pass

def find_closest_match(words, target):
    # print(f'try to find "{target}" in "{words}"')
    target_words = nltk.word_tokenize(target)
    similarities = []
    for i in range(len(words) - len(target_words) + 1):
        similarity = nltk.jaccard_distance(set(target_words), set(words[i:i+len(target_words)]))
        # print(f'similarity between {target_words} and {words[i:i+len(target_words)]} : {similarity}')
        similarities.append(similarity)
    # print(similarities)
    # print(type(similarities))
    best_match_index = similarities.index(min(similarities))

    return ' '.join(words[best_match_index:best_match_index+len(target_words)])
def get_sentence_timings(transcription, sentence):
    """
    Returns the start and end timings of a sentence in a transcription object.

    Args:
    transcription (Transcription): The transcription object.
    sentence (str): The sentence to search for.

    Returns:
    tuple: A tuple containing the start and end timings of the sentence.
    """
    word_list = [d['word'] for d in transcription.words]
    sentence = find_closest_match(word_list, sentence)
    # sentence = onlychars(sentence)
    # print(f'try to find {sentence} in {word_list}')
    
    words = transcription.words
    sentence = sentence.replace('...', ' ').replace('\"', ' ').replace('.', '')
    sentence_words = sentence.split()
    sentence_start = None
    sentence_end = None

    # print(f'try to find {sentence_words} in {words}')
    
    i = 0
    while i < len(words):
        if words[i]['word'].lower() == sentence_words[0].lower():
            # print(f'found {sentence_words[0]}')
            j = 1
            while j < len(sentence_words) and i + j < len(words) and words[i + j]['word'].lower() == sentence_words[j].lower():
                j += 1
            if j == len(sentence_words):
                sentence_start = words[i]['start']
                sentence_end = words[i + j - 1]['end']
                break
            # print(f'found {sentence_words[:j]} but not the rest')
        i += 1
        
    if(sentence_start == None or sentence_end == None):
        print(f"no timings found for '{sentence}'")
    else:
        print(f"timings found for '{sentence}' : {sentence_start} to {sentence_end}")    
        sentence_start = round(sentence_start, 2)
        sentence_end = round(sentence_end, 2)
    return sentence_start, sentence_end
def create_all_timings(timing_description, descriptions):
    timings = []
    m=0
    for key in list(descriptions.keys()):
        # timing_description is the original transcription
        # key is the sentence to look online for
        start_time, end_time = get_sentence_timings(timing_description, key)
        if(start_time < m):
            start_time = m
        m = end_time
        if(start_time == None or end_time == None):
            continue
        timings.append((start_time, end_time))
    return timings
def select_images(directory):
    file_list = os.listdir(directory)
    print(file_list)
    file_list = [f for f in file_list if os.path.isfile(os.path.join(directory, f))]
    file_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    file_paths = [os.path.join(directory, f) for f in file_list]
    return file_paths

def color_screen_to_video(color = [0,0,0], format = (1080, 1920), duration = 60) -> VideoClip:
    image = np.full((*format, 3), color, dtype=np.uint8)
    clip = ImageClip(image)
    clip = clip.set_duration(duration)
    return clip
def image_to_video(image_path, duration = 60) -> VideoClip:
    image = ImageClip(image_path)
    clip = image.set_duration(duration)
    return clip
def add_audio_to_video(video, audio_path) -> VideoClip:
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)
    return video
def add_multiple_images_to_video(input_video:VideoClip, images, timings, write_video = False, write_path = None) -> VideoClip:
    print(f"Adding {len(images)} images to the video")
    print(f"Images : {images}")
    print(f"Timings : {timings}")
    
    images = images.copy()
    clip = input_video
    clips = []
    offset = 0
    
    for start_time, end_time in timings:
        if start_time == None or end_time == None:
            print("Could not find timings")
            continue
        print(f"Adding image {images[0]} from {start_time} to {end_time}")
        # print(len(images))
        clip_before = clip.subclip(0, start_time-offset)
        clip_with_image = clip.subclip(start_time-offset, end_time-offset)
        clip_after = clip.subclip(end_time-offset, clip.duration)
        offset += clip_before.duration + clip_with_image.duration
        
        try:
            image_clip = ImageClip(images.pop(0))
            image_clip = image_clip.resize(width=clip.w/2, height=clip.h/2)
            image_clip = image_clip.set_pos(('center', 'center'))
            image_clip = image_clip.set_start(0).set_duration(clip_with_image.duration)
        except Exception as e:
            print(f"Error while adding image : {e}")
            # clips.append(clip_with_image)
            continue
        final_clip_middle = CompositeVideoClip([clip_with_image, image_clip])

        print(f"Clip before duration : {clip_before.duration}")
        print(f"Final clip middle duration : {final_clip_middle.duration}")
        
        final_clip = concatenate_videoclips([clip_before, final_clip_middle]) # if clip_before.duration > 0 else final_clip_middle

        clip = clip_after
        
        clips.append(final_clip)

    clips.append(clip)
    final_clip = concatenate_videoclips(clips, method="chain")

    if(write_video):
        print('writing temporary video')
        if(write_path == None):
            write_path = 'output-temp.mp4'
        final_clip.write_videofile(write_path.replace('.mp4', '') + '.mp4')
        return VideoFileClip(write_path)
        
    return final_clip
def add_subtitles_to_video(video, subtitles, position = ('center', 0.9)) -> VideoClip:
    # Create a list of TextClip objects for each subtitle
    text_clips = []
    for word in subtitles.words:
    # for word in subtitles['words']:
        # print(word['word'], word['start'], word['end'])
        text_clip = TextClip(word['word'], fontsize=60, color='white')
        text_clip = text_clip.set_start(word['start']).set_duration(word['end'] - word['start'])
        text_clip = text_clip.set_position(position)
        text_clips.append(text_clip)

    # Overlay the text clips on the video
    final_video = CompositeVideoClip([video] + text_clips)
    
    return final_video

def getTextTimingsOfMp4(mp4file):
    # Create a temporary MP3 file
    mp3file = mp4file.replace('.mp4', '.mp3')

    # Load the video clip and extract the audio
    video = VideoFileClip(mp4file)
    audio = video.audio
    audio.write_audiofile(mp3file)

    # Close the video and audio objects
    video.close()
    audio.close()
    del video
    del audio


    # Create an OpenAI client
    global client
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY_LORRAIN'))

    # Open the MP3 file and transcribe it
    with open(mp3file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )

    # Delete the temporary MP3 file
    # os.remove(mp3file)

    return transcript


def create_video_from_text(text, video_name = None, background_video_path = 'videos\\satifying\\satisfying.mp4', output_name = None):
    if(video_name == None):
        video_name = onlychars(" ".join(text.split(' ')[0:5]))
    if(output_name == None):
        output_name = video_name.replace(' ', '-') + '.mp4'
        
    print(f"Starting creation of the video : {video_name}")
    
    # create and clear the folder for the video
    print("Creating folder")
    local_path = os.path.join('Projects', video_name)
    if(not os.path.exists(local_path)):
        os.mkdir(local_path)
    clear_folder(local_path)

    # create voice for the video
    print(f"Creating voice")
    voice_path = os.path.join(local_path, 'audio\\voice.mp3')
    if(not os.path.exists(os.path.join(local_path, 'audio'))):
        os.mkdir(os.path.join(local_path, 'audio'))
    # print(f"Voice path : {voice_path}")
    generate_voice(text, voice_path)
    timing_description = getTextTimingsOfMp3(voice_path)
    final_text  = timing_description.text
    print(timing_description.duration)
    
    # Find Images for the video
    print("Finding images")
    descriptions = get_pictures(final_text)
    print(f"Descriptions : {descriptions}")
    images_path = os.path.join(local_path, 'images')
    if(not os.path.exists(images_path)):
        os.mkdir(images_path)
    download_images(list(descriptions.values()), images_path)
    convert_folder_to_png(images_path)
    images = select_images(images_path)
    
    # Find timings for images
    print("Finding timings")
    timings = create_all_timings(timing_description, descriptions)
            
    # create whole video
    print("Creating video")
    # video = image_to_video(image_path=background, duration=timing_description.duration)
    video = VideoFileClip(background_video_path)
    video = video.subclip(0, timing_description.duration)
    new_width_ratio = 1/3
    new_width = int(video.w * new_width_ratio)
    start_x = video.w - 2*new_width
    video = video.crop(x1=start_x, width=new_width)
    print(video.size)
    # video = video.resize(width=1080, height=1920)
    print('Done')
    print(f"Adding images to video")
    video = add_multiple_images_to_video(video, images, timings, write_video=True, write_path=os.path.join(local_path,'temp.mp4'))
    print('Done')
    print(f"Adding subtitles to video")
    video = add_subtitles_to_video(video, timing_description)
    print('Done')
    print(f"Adding voice to video")
    video = add_audio_to_video(video, voice_path)
    print('Done')
    
    print(f"Writing video to {output_name}")   
    video.write_videofile(os.path.join(local_path, output_name), fps=24)
    print('Done')
    return video

def create_video_from_video(video_path, video_name = None, output_name = None):
    if(video_name == None):
        video_name = video_path.split('\\')[-1].split('.')[0]
    print(f"Starting creation of the video : {video_name}")
    if(output_name == None):
        output_name = video_name.replace(' ', '-') + '.mp4'
    
    # create and clear the folder for the video
    print("Creating folder")
    local_path = os.path.join('Projects', video_name)
    if(not os.path.exists(local_path)):
        os.mkdir(local_path)
    clear_folder(local_path)
    
    # create transcript of the video
    print(f"Creating transcript")
    timing_description = getTextTimingsOfMp4(video_path)
    final_text  = timing_description.text
    
    # Find Images for the video
    print("Finding images")
    descriptions = get_pictures(final_text)
    print(f"Descriptions : {descriptions}")
    images_path = os.path.join(local_path, 'images')
    if(not os.path.exists(images_path)):
        os.mkdir(images_path)
    download_images(list(descriptions.values()), images_path)
    convert_folder_to_png(images_path)
    images = select_images(images_path)
    
    # Find timings for images
    print("Finding timings")
    timings = create_all_timings(timing_description, descriptions)
            
    # create whole video
    video = VideoFileClip(video_path)
    print(f"Adding images to video")
    video = add_multiple_images_to_video(video, images, timings)
    print('Done')
    print(f"Adding subtitles to video")
    video = add_subtitles_to_video(video, timing_description)
    print('Done')
    
    print(f"Writing video to {output_name}")   
    video.write_videofile(os.path.join(local_path, output_name), fps=24)
    print('Done')
    return video

if __name__ == '__main__':
    text = """
            "Houdini" is the lead single of Eminem?s twelfth studio album The Death of Slim Shady (Coup de Gr?ce). The track is named after Harry Houdini, a popular magician known for his death-defying stunts like the Chinese Water Torture Cell, which Eminem replicates in the intro to his 1999 track, Role Model.
           """
    text2 = """
    Once upon a time, in a world where books could talk, there was a small, old book named "Whispers of Wisdom". Unlike the newer, flashier books, Whispers was often overlooked. However, it had a secret power: it could change its content based on the reader's needs.

One day, a young girl named Lily, who was struggling with self-doubt, found Whispers in a dusty corner of the library. As she opened it, the pages fluttered and the words rearranged themselves. She began to read, and to her surprise, the book was all about overcoming self-doubt and believing in oneself.

With each visit to the library, Lily found new wisdom in Whispers. It became her guiding light, helping her navigate through life's challenges. The once overlooked book had found its purpose, and Lily had found her confidence.

And so, Whispers of Wisdom continued to share its magic, reminding everyone that sometimes, the most unassuming things hold the greatest treasures."""
    text3 = """
    Once upon a time, in a sunny little town, lived a curious cat named Momo. Momo had a unique fascination - she loved butterflies. Their vibrant colors and delicate wings captivated her, and she dreamt of playing with them.

One day, Momo saw a group of butterflies dancing in the garden. She approached slowly, trying not to scare them. But every time she got close, they fluttered away. Momo was disappointed but didn't give up.

She observed the butterflies and noticed they loved flowers. So, Momo decided to learn about flowers. She watched as bees collected nectar, and birds pecked at petals. She learned which flowers bloomed when and where the butterflies liked to rest.

With her newfound knowledge, Momo started spending her days in the flower patches, waiting patiently. One sunny afternoon, a butterfly landed on a flower next to her. Momo held her breath, and to her delight, the butterfly didn't fly away. Instead, it fluttered its wings, as if inviting Momo to play.

From that day forward, Momo became a friend to the butterflies. She played with them among the flowers, chasing their delicate dance but never trying to catch them. Momo's patience and understanding had turned her dream into reality, proving that sometimes, the best way to play is to let the game come to you."""
    text4 = """ I was a humble park bench, nestled under the shade of an ancient oak tree. I've seen countless sunrises, each one painting the sky with a different palette of colors. One day, a young girl named Lily sat on me, her eyes filled with curiosity. She visited me every day, her laughter echoing through the park, her stories bringing me to life. Years passed, and Lily grew older. Her visits became less frequent, but her smile never faded. One day, she returned, not as a visitor, but as the park's new caretaker. She sat on me, her eyes filled with memories. "Hello, old friend," she whispered, and I felt a warmth that the sun could never provide. """
    text5 = """Once upon a time, in the heart of bustling New York City, I found a peculiar little shop. It was tucked away in a narrow alley, almost invisible to the unobservant eye. The sign above the door read "Time Emporium."Intrigued, I stepped inside. The shop was filled with clocks of all shapes and sizes, each ticking in its own unique rhythm. An old man, with a white beard and twinkling eyes, greeted me. He introduced himself as the Timekeeper.He showed me a small, antique pocket watch. "This one," he said, "holds the power to slow down time." I laughed, thinking it was a joke, but he insisted it was true.Curious, I bought the pocket watch and left the shop. As I stepped back into the bustling city, I opened the watch. Suddenly, the world around me slowed. People moved like they were wading through honey, cars inched forward, and the city's noise faded into a soft hum.I marveled at the pocket watch, feeling like I had the world to myself. But then I remembered, time waits for no one. I closed the watch, and the city sprang back to life.I still visit the Time Emporium, and the old Timekeeper always has a new story to tell. But that's a tale for another time."""
    
    create_video_from_text(text5, 'NYC-clock', 'videos\\brackgrounds\\yellow_matrix.mp4')
