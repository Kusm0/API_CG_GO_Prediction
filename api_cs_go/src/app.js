import $ from 'jquery'
import {api} from './api/base.js'

let teams;
let target_el;

let teams_search = $('.teams_list_search');
let teams_select = $('.teams_list');
let teams_list = $('#Teams');

teams_select.find('.close').off('click').on('click', function (e) {
    e.preventDefault();
    e.stopPropagation();
    teams_search.val('');
    teams_search.trigger('input');
    teams_select.addClass('hidden');
})

teams_search.off('input').on('input', function (e) {
    e.preventDefault();
    e.stopPropagation();
    const query = $(this).val().toLowerCase();
    if (query === '' && query?.length === 0) {
        teams_list.find('.team').removeClass('hidden');
        return;
    }
    teams_list.find('.team').each(function () {
        const teamId = $(this).data('team-id').toString();
        const teamName = $(this).data('team-name').toString().toLowerCase();

        if (teamId.includes(query) || teamName.includes(query)) {
            $(this).removeClass('hidden');
        } else {
            $(this).addClass('hidden');
        }
    });
})

function handleItemClick(e) {
    e.preventDefault();
    e.stopPropagation();
    $(target_el).find('.clear').trigger('click')
    $(target_el).contents().filter(function () {
        return this.nodeType === Node.TEXT_NODE;
    }).remove();
    $(target_el).append($(this).data('team-name'));
    $(target_el).attr('data-team-id', $(this).data('team-id'));
    $(this).addClass('superhidden');
    $(target_el).removeClass('error_unset');
    teams_search.val('');
    teams_search.trigger('input');
    teams_select.addClass('hidden');
}

$(async function () {
    teams = (await api.get('/get_teams')).data;
    teams.forEach((team) => {
        let team_el = $(`
            <div class="team" data-team-id="${team.ID}" data-team-name="${team.Name}">
                <span class="team_name">${team.Name}</span>
                <span class="team_id">${team.ID}</span>
            </div>
        `)
            .off('click')
            .on('click', handleItemClick);
        teams_list.append(team_el);
    })
    $('#numberOfTeams').trigger('change');
})

const verbose = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
}

function handleTeamClick(e) {
    target_el = this;
    e.preventDefault();
    e.stopPropagation();
    teams_search.val('');
    teams_search.trigger('input');

    const popupWidth = $(teams_select)[0].offsetWidth;
    const popupHeight = $(teams_select)[0].offsetHeight;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Get mouse coordinates adjusted for scroll
    let mouseX = e.clientX + window.scrollX; // Adjust for horizontal scroll
    let mouseY = e.clientY + window.scrollY; // Adjust for vertical scroll

    console.log(mouseX, popupWidth, viewportWidth);

    // Prevent the popup from overflowing beyond the viewport
    if (mouseX + popupWidth > window.scrollX + viewportWidth) {
        mouseX = window.scrollX + viewportWidth - popupWidth - 50;
    }
    if (mouseY + popupHeight > window.scrollY + viewportHeight) {
        mouseY = window.scrollY + viewportHeight - popupHeight - 50;
    }

    // Set popup position
    teams_select.css('left', `${mouseX}px`);
    teams_select.css('top', `${mouseY}px`);
    teams_select.removeClass('hidden');
}

let left_side = $('.split-one');
let right_side = $('.split-two');
let champion = $('.champion');

$('.round-one .team').off('click').on('click', handleTeamClick);

let match = $(`
                <ul class="matchup">
                    <li class="team team-top">
                        <div class="clear">
                            <svg clip-rule="evenodd" fill="#000000" fill-rule="evenodd" stroke-linejoin="round" stroke-miterlimit="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="m12 10.93 5.719-5.72c.146-.146.339-.219.531-.219.404 0 .75.324.75.749 0 .193-.073.385-.219.532l-5.72 5.719 5.719 5.719c.147.147.22.339.22.531 0 .427-.349.75-.75.75-.192 0-.385-.073-.531-.219l-5.719-5.719-5.719 5.719c-.146.146-.339.219-.531.219-.401 0-.75-.323-.75-.75 0-.192.073-.384.22-.531l5.719-5.719-5.72-5.719c-.146-.147-.219-.339-.219-.532 0-.425.346-.749.75-.749.192 0 .385.073.531.219z"/></svg>
                        </div>
                    </li>
                    <li class="team team-bottom">
                        <div class="clear">
                            <svg clip-rule="evenodd" fill="#000000" fill-rule="evenodd" stroke-linejoin="round" stroke-miterlimit="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="m12 10.93 5.719-5.72c.146-.146.339-.219.531-.219.404 0 .75.324.75.749 0 .193-.073.385-.219.532l-5.72 5.719 5.719 5.719c.147.147.22.339.22.531 0 .427-.349.75-.75.75-.192 0-.385-.073-.531-.219l-5.719-5.719-5.719 5.719c-.146.146-.339.219-.531.219-.401 0-.75-.323-.75-.75 0-.192.073-.384.22-.531l5.719-5.719-5.72-5.719c-.146-.147-.219-.339-.219-.532 0-.425.346-.749.75-.749.192 0 .385.073.531.219z"/></svg>
                        </div>
                    </li>
                </ul>
            `);

match.find('.team').off('click').on('click', handleTeamClick);
match.find('.clear').off('click').on('click', function (e) {
    e.preventDefault();
    e.stopPropagation();
    let parent = $(this).parent();
    teams_list.find(`.team[data-team-id="${parent.attr('data-team-id')}"]`).removeClass('superhidden');
    parent.removeAttr('data-team-id');
    parent.removeClass('error_unset');
    parent.removeClass('winner');
    parent.removeClass('loser');
    parent.contents().filter(function () {
        return this.nodeType === Node.TEXT_NODE;
    }).remove();
});

$('#numberOfTeams').off('change').on('change', function (e) {
    e.preventDefault();
    e.stopPropagation();
    left_side.empty();
    right_side.empty();
    champion.empty();
    let number_of_teams = $(this).val();
    let one_side = number_of_teams / 2;
    let brackets = one_side / 2;
    let rounds = Math.log2(brackets);

    for (let i = 1; i < rounds + 2; i++) {

        let round_el_left = $(`
            <div class="round round-${verbose[i]} current">
                <div class="round-details">Round ${i}</div>
            </div>
        `);
        let round_el_right = round_el_left.clone();

        for (let j = 0; j < brackets; j++) {
            let clone1 = match.clone(true)
            let clone2 = match.clone(true)
            if (i !== 1){
                clone1.find('.team').off('click');
                clone2.find('.team').off('click');
            }
            round_el_left.append(clone1);
            round_el_right.append(clone2);
        }
        left_side.append(round_el_left);
        right_side.prepend(round_el_right);
        brackets /= 2;
    }
    let round_el_final = $(`
        <div class="final current">
            <i class="fa fa-trophy"></i>
            <div class="round-details">championship</div>
        </div>
    `);

    let clone = match.clone(true);
    clone.find('.team').off('click');
    clone.addClass('championship');
    round_el_final.append(clone);
    champion.append(round_el_final);
})

$('.start_tournament').off('click').on('click', async function (e) {
    e.preventDefault();
    e.stopPropagation();
    let unset_ids = $('.round-one .matchup .team:not([data-team-id])');
    let unset_ids_arr = unset_ids.get();
    if (!unset_ids_arr || unset_ids_arr?.length > 0){
        unset_ids.addClass('error_unset');
        return;
    }
    let ids = $('.round-one .matchup .team[data-team-id]').map(function () {
        return parseInt($(this).attr('data-team-id'));
    }).get();
    let result = (await api.post('/simulate_tournament', {teams: ids})).data;
    let matches = result['matches'];
    let rounds = Object.keys(matches).length;
    console.log(result, rounds)
    $('.clear_tournament').trigger('click')
    for (let i = 1; i < rounds+1; i++) {
        let matchups;
        if (i===rounds){
            matchups = $(`.final .matchup`);
        } else {
            matchups = $(`.round-${verbose[i]} .matchup`);
        }
        let round = matches[`round_${i}`];
        console.log(round)
        for (let j = 0; j < round.length; j++) {
            let matchup = matchups.get(j);
            let team1 = $(matchup).find('.team-top');
            let team2 = $(matchup).find('.team-bottom');
            let team1_winner = round[j]['team1_id']===round[j]['winner_id'];
            $(team1).contents().filter(function () {
                return this.nodeType === Node.TEXT_NODE;
            }).remove();
            $(team1).append(round[j]['team1']);
            $(team1).addClass(team1_winner?'winner':'loser');
            $(team2).contents().filter(function () {
                return this.nodeType === Node.TEXT_NODE;
            }).remove();
            $(team2).append(round[j]['team2']);
            $(team2).addClass(team1_winner?'loser':'winner');

            if (i === 1){
                $(team1).attr('data-team-id', round[j]['team1_id']);
                $(team2).attr('data-team-id', round[j]['team2_id']);
            }
        }
    }
})

$('.random_tournament').off('click').on('click', function (e) {
    e.preventDefault();
    e.stopPropagation();

    let n = parseInt($('#numberOfTeams').val());
    let random_teams = [];
    const used_indices = new Set();

    while (random_teams.length < n) {
        const random_index = Math.floor(Math.random() * teams.length);
        if (!used_indices.has(random_index)) {
            used_indices.add(random_index);
            random_teams.push(teams[random_index]);
        }
    }

    $('.clear_tournament').trigger('click')

    $('.round-one .matchup .team').each(function (index, element) {
        $(element).contents().filter(function () {
            return this.nodeType === Node.TEXT_NODE;
        }).remove()
        $(element).append(random_teams[index]['Name'])
        $(element).attr('data-team-id', random_teams[index]['ID'])
        teams_list.find(`.team[data-team-id="${random_teams[index]['ID']}"]`).addClass('superhidden')
    })
})

$('.clear_tournament').off('click').on('click', function (e) {
    e.preventDefault();
    e.stopPropagation();
    $('.round .matchup .team .clear').trigger('click')
    $('.final .matchup .team .clear').trigger('click')
})