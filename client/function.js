
RESOURCE_URL = './frame'
function UpserverFrame(image){
    fetch(RESOURCE_URL, {
        method: 'POST',
        headers: {
            accept: 'application.json',
            'content-type': 'application/json'
        },
        body: image,
        caches: 'no-cache'
    })
    .then(response => response.json())
    .then(data => ModelResponseHandler(data))
    .catch(error => console.log(error))
}

function BufferFrame(){
    this.image = undefined;
    this.isSelected = false;
    this.isProccessed = false;
}

function Buffer(length, idMaxPoint){
    this.listFrames = []
    for (let i=0; i<length; i++){
        this.listFrames.push(new BufferFrame())
    }
    this.idPoint = 0
    this.idMaxPoint = idMaxPoint
    this.idLastProccessed = -1
    this.idNextProccessed = 0

    this.Expired = function(){
        // Pop first element and push new init element to tail
        let expiredFrame = this.listFrames.shift()
        this.listFrames.push(new BufferFrame())
        
        this.idLastProccessed -= 1
        this.idNextProccessed -= 1
        this.idPoint -= 1

        if (expiredFrame.isSelected === true){
            savedFrames.push(expiredFrame.image)
        }

        delete expiredFrame
    }

    this.LabelFrames = function(action, indicesNeighbor){
        this.idNextProccessed = this.idLastProccessed + action
        // 2 lines below may not be used
        this.listFrames[this.idNextProccessed].isProccessed = true
        this.listFrames[this.idNextProccessed].isSelected = true

        for(let idNeighbor of indicesNeighbor){
            this.listFrames[idNeighbor].isSelected = true
        }
    }

    this.CookieFrame = function(image){
        this.listFrames[this.idPoint].image = image
        this.idPoint += 1

        if (this.idNextProccessed <= this.idPoint - 1){
            UpserverFrame(this.listFrames[this.idNextProccessed].image)
            this.idLastProccessed = this.idNextProccessed
            this.idNextProccessed = Infinity
        }

        if (this.idPoint >= this.idMaxPoint){
            this.Expired()
        }
    }
}

function ModelResponseHandler(response){
    let action = response.get('action')
    let indicesNeighbor = response.get('indices neighbor')
    buffer.LabelFrames(action, indicesNeighbor)
}

// var savedFrames = []


// alert(Buffer)
// console.log(typeof(Buffer))
// console.log(Buffer)
