const inputContainer = new InputContainer()
const videoWorks = !!document.createElement('video').canPlayType;
if (videoWorks) {
    inputContainer.video.controls = false;
    inputContainer.videoControls.classList.remove('hidden');
}

const outputContainer = new OutputContainer()
// const videoWorks = !!document.createElement('video').canPlayType;
if (true) {
    // outputContainer.video.controls = false;
    outputContainer.videoControls.classList.remove('hidden');
}

// var buffer = new Buffer(length=120, idMaxPoint=60)
// alert(outputContainer)
// var mediaRecorder;
// b = new MediaSource()
// Add functions here
// Get access to the camera!
// exports.blobing;
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({audio: false, 
        video: true
        // video: {
        //     mandatory: {
        //     minWidth: 768,
        //     minHeight: 768,
        //     minFrameRate: 30
        // }}
    }).then(function(stream) {
        inputContainer.video.srcObject = stream;
        // video.src = window.URL.createObjectURL(stream)
        inputContainer.video.play();
        inputContainer.video.onloadedmetadata = () => {
            outputContainer.video.width = inputContainer.video.clientWidth;
            outputContainer.video.height = inputContainer.video.clientHeight;
            captureBlob = setInterval(() => {
                if(inputContainer.video.paused){}
                else{
                    // alert('Thay doi kich thuoc');
                    // outputContainer.video.width = inputContainer.video.clientWidth;
                    // outputContainer.video.height = inputContainer.video.clientHeight;
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = outputContainer.video.width;
                    tempCanvas.height = outputContainer.video.height;
                    const tempRCanvas = tempCanvas.getContext('2d');
                    tempRCanvas.drawImage(inputContainer.video, 0, 0, outputContainer.video.width, outputContainer.video.height);
                    img = tempRCanvas.getImageData(0, 0, outputContainer.video.width, outputContainer.video.height);
                    outputContainer.listImage.push(img);
                    outputContainer.fcUpdateVideoDuration();
                }
                
            }, 33);
        };
        
        // mediaRecorder = new MediaRecorder(stream);
        // // let chunks = [];//Cria uma matriz para receber as parte.
        // mediaRecorder.ondataavailable = (data) => {
        //     console.log('recorder available');
        //     // chunks.push(data.data)//Vai adicionando as partes na matriz
        //     blobing = new Blob(data.data, {type: 'image/png'});
        //     console.log(num_show_blob, blob);
        // };
        // mediaRecorder.onstop = () => {//Quando ativar a função parar a gravação
        // //Cria o BLOB com as partes acionadas na Matriz
        //     // const blob = new Blob(chunks, { type: 'audio/wav' });
        //     // alert('Media Recorder Stopped', blob)
        // }
    });
}



document.addEventListener('DOMContentLoaded', () => {
    if (!('pictureInPictureEnabled' in document)) {
        inputContainer.pipButton.classList.add('hidden');
    }
});