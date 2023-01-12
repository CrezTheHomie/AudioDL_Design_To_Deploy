## About

In this repo, we are going to learn how to Dockerize NGINX webserver and uWSGI to flask to TF package.  
Not only that, we will deploy to AWS instance.

[Playlist link](https://www.youtube.com/playlist?list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp)

## Notes

### Video One  
NGINX is the leading web server, but it has trouble parsing python environments.  
uWSGI can handle python evironments.   
Question: Why not just use uWSGI?  .
### Video Two  
Grabbed the speech commands dataset from [Google's AI Blog](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGlGeDBadHB6UjNpV09Cd211QkhsZW13a3JJQXxBQ3Jtc0tsNzVHRktjbEptUVJic2pIUlVyUzlhbElMVUF2Y25fRWVJLTh4N0I0Rk54UmtEbVR6YXVhVzlzY3pwak93OGVxLXpsblhnbWJNalJ2cWlLMk9GN0MtalNCOVFPU1N5MlVkTTNFMWJSaWRabE9pN3QzTQ&q=https%3A%2F%2Fai.googleblog.com%2F2017%2F08%2Flaunching-speech-commands-dataset.html&v=VPJ2jazh_KI)  
Perpared the data dictionary by processing and creating a JSON with mappings, labels, MFCCs, and associated file names.  
Storing this data dictionary as JSON because its faster to process this and then train the model, rather than processing MFCCs and such each time model is trained.
### Video Three
Our model is a CNN to learn from the MFCCs of the single word utterances.  
Saved model is "model.h5"
### Video Four
Define and implement a service for taking an audio file and passing it to the model for prediction.
### Video Five
This is where I deviated from Valerio. Flask API is easy to understand and work with. REST APIs are pretty straightforward. However, dockerizing the uWSGI webserver was something I just couldn't follow with because uWSGI package from pip exits with error. Switched to BentoML. 

### Non-video notes
Not the best documentation for passing a file to BentoML, but using the service API response to general a curl request, I figured out the post request wants an application/octet-stream rather than a multipart/form-data.  
This makes sense because it should pass the filestream along the post request. I still don't fully understand what multipart/form-data would have needed. I do understand multipart/form-data accepts multiple different payloads in one request (such as multiple files).

After watching [this video](https://www.youtube.com/watch?v=vqFbyVu7_dY), it seems multipart form data was the original way to pass along chuncks of data. When curl was used more, it made it easier by having -F as an arg to pass instead of delimiting chuncks with the random string.  
Ex: curl -F person=anonymous -F secret=@file.txt http://example.com/submit.cgi

[Note] Interestingly, BentoML does not support directly passing audio data. It does directly support passing image data.  