function updateData() {
    $.getJSON('/data', function(data) {
        $('#team1_odds').text(data.team1_odds);
        $('#team2_odds').text(data.team2_odds);
        $('#team_assignment').text(data.team_assignment);
        $('#timer').text(data.timer);
        $('#team_1_score').text(data.team_1_score);
        $('#team_2_score').text(data.team_2_score);
        $('#frame').text(data.frame);

        let playerInfoTeam1 = '';
        let playerInfoTeam2 = '';

        for (let i = 0; i < 10; i++) {
            const status = data.player_statuses[i];
            const statusClass = status.toLowerCase() === 'dead' ? 'dead-player' : '';
            const hpBarClass = status.toLowerCase() === 'dead' ? 'hp-bar-dead' : 'hp-bar';
            const playerInfo = `
                <li>
                    <span class="player-status ${statusClass}">Player ${i+1}: ${status}</span>
                    <div class="player-hp">
                        <div class="hp-bar-container">
                            <div class="${hpBarClass}" style="width: ${data.hp[i]}%;"></div>
                            <span class="hp-text">HP: ${data.hp[i]}</span>
                        </div>
                    </div>
                    <span class="player-weapon">${data.weapons[i]}</span>
                </li>`;
            
            if (i < 5) {
                playerInfoTeam1 += playerInfo;
            } else {
                playerInfoTeam2 += playerInfo;
            }
        }

        $('#player_info_team1').html(playerInfoTeam1);
        $('#player_info_team2').html(playerInfoTeam2);

        if (data.team_assignment === "Team 0 is CT, Team 1 is T") {
            $('#team1').addClass('blue-background').removeClass('sand-background');
            $('#team2').addClass('sand-background').removeClass('blue-background');
        } else if (data.team_assignment === "Team 0 is T, Team 1 is CT"){
            $('#team2').addClass('blue-background').removeClass('sand-background');
            $('#team1').addClass('sand-background').removeClass('blue-background');
        }

        // Check if it's a replay
        if (data.is_replay) {
            $('#replay-indicator').show();
        } else {
            $('#replay-indicator').hide();
        }
    });
}

// Update data every second
setInterval(updateData, 1000);

// Initial update when the page loads
$(document).ready(function() {
    updateData();
});