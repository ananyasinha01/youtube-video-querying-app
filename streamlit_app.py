import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

# Load environment variables
load_dotenv()

# Set up Groq API
groq_api_key = os.getenv("GROQ_API_KEY")  # Get the Groq API key from the environment
client = Groq(api_key=groq_api_key)

# Proxy configuration
proxy_url = "http://192.168.29.67:8080"  # This proxy points to your droplet or another proxy configuration

# Prompt for summarization
summary_prompt_template = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video, providing the important points within 250 words.
Please summarize the text given here: """

# Function to get transcript from YouTube video with proxy configuration
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        # Use proxy to avoid getting blocked by YouTube
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id, proxies={"https": proxy_url})

        return transcript_text  # Return the list of dictionaries directly
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

# Function to generate summary using Groq API
def generate_groq_summary(text_chunk, prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt + text_chunk}],
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

# Function to create embeddings for chunks
def create_embeddings(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return np.array(embedding_model.embed_documents(chunks)).astype('float32')

# Function to create transcript chunks with overlap
def create_transcript_chunks(transcript, block_size=30, overlap_size=5):
    chunks = []
    i = 0
    
    while i < len(transcript):
        # Get the start time of the first block in the current chunk
        start_time = transcript[i]['start']
        
        # Create the chunk by concatenating text from the current block and the next 'block_size - 1' blocks
        chunk_text = " ".join([transcript[j]['text'] for j in range(i, min(i + block_size, len(transcript)))])
        
        # Store the chunk and its start time
        chunks.append({
            'chunk_text': chunk_text,
            'start_time': start_time
        })
        
        # Move to the next block with overlap
        i += block_size - overlap_size
    
    return chunks

# Streamlit app setup
st.title("YouTube Video Summarizer and Querying")

# Input for YouTube link
youtube_link = st.text_input("Enter YouTube Video Link:")

# Display video thumbnail
if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

# Button for summarizing the video
if st.button("Get Summarized Notes"):
    transcript_text = extract_transcript_details(youtube_link)

    if transcript_text:
        st.write("Transcript Extracted Successfully!")

        # Split transcript into chunks for summarization
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(" ".join([entry['text'] for entry in transcript_text]))  # Join text for summarization
        st.write(f"Number of Chunks Created for Summarization: {len(chunks)}")

        # Create embeddings for chunks
        chunk_embeddings = create_embeddings(chunks)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = generate_groq_summary(chunk, summary_prompt_template)
            summaries.append(summary)

        # Combine summaries and generate the final summary
        combined_summary_text = " ".join(summaries)
        final_summary = generate_groq_summary(combined_summary_text, summary_prompt_template)
        st.markdown("## Final Summary:")
        st.write(final_summary)
    else:
        st.error("Failed to extract transcript.")

# Button for querying
st.markdown("## Ask a question about the video:")
question = st.text_input("Enter your question:")

if st.button("Submit Question"):
    if question.strip():  # This checks if the question is not empty or just white spaces
        # Fetch transcript text again
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            # Create chunks for querying
            transcript_chunks = create_transcript_chunks(transcript_text)

            # Create embeddings for the chunks
            chunk_embeddings = create_embeddings([chunk['chunk_text'] for chunk in transcript_chunks])

            # Create a FAISS index for querying
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
            index.add(chunk_embeddings)

            # Generate embedding for the question
            question_embedding = create_embeddings([question])[0]

            # Search for the nearest neighbors in the FAISS index
            k = 5  # Number of nearest neighbors to retrieve
            D, I = index.search(np.array([question_embedding]).astype('float32'), k)

            # Retrieve the corresponding chunks and their start times
            retrieved_chunks = [transcript_chunks[i] for i in I[0]]

            # Create a combined prompt with retrieved chunks
            combined_prompt = f"You are an expert on the contents of the YouTube video transcript. If the question asked is not relevant to the video, simply say 'No relevant information found in the transcript.'. Answer concisely based on the following information:\n"
            for chunk in retrieved_chunks:
                combined_prompt += f"{chunk['chunk_text']}\n"

            combined_prompt += f"Question: {question}\nAnswer: "

            # Get the answer using Groq API
            final_answer = generate_groq_summary(combined_prompt, "Answer concisely, within 300 words.")

            # Display the query result
            if final_answer.strip():  # Check if the answer is not empty
                st.markdown("## Answer to your question:")
                st.write(final_answer)

                # Display the start time of the chunk with the highest similarity
                formatted_time = f"{int(retrieved_chunks[0]['start_time'] // 60):02}:{int(retrieved_chunks[0]['start_time'] % 60):02}"
                st.write(f"Check out this timestamp: {formatted_time}")
            else:
                st.warning("No relevant information found in the transcript.")
        else:
            st.error("Failed to extract transcript.")
    else:
        st.warning("Please enter a valid question before submitting.")  # This will display if the question is empty or only white spaces
