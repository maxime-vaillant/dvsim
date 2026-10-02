// Browser download of a JSON array of records, one record per line (compact but still diff-friendly).

export function downloadRecords(filename, records) {
    const text = '[\n' + records.map(record => '  ' + JSON.stringify(record)).join(',\n') + '\n]\n';
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
