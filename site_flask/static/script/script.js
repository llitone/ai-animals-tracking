const folderInput = document.getElementById('folderInput');

folderInput.addEventListener('change', () => {
    result = [];
    const folderFiles = folderInput.files;
    for (const file of folderFiles) {
        result.push("C:/" + file.webkitRelativePath);
    }
    console.log(result);
    fetch('http://127.0.0.1:5000/report', { // ???
        method: 'POST',
        mode: 'no-cors',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(result, null, 2)
    })
    .then(response => response)
    .then(data => console.log(data))
    .catch(error => console.error('Произошла ошибка:', error));
});

