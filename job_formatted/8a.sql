SELECT min(an1.name) AS actress_pseudonym,
       min(t.title) AS japanese_movie_dubbed
FROM aka_name AS an1,
     cast_info AS ci,
     company_name AS cn,
     movie_companies AS mc,
     name AS n1,
     role_type AS rt,
     title AS t
WHERE ci.note ='(voice: English version)'
    AND cn.country_code ='[jp]'
    AND mc.note like '%(Japan)%'
    AND mc.note not like '%(USA)%'
    AND n1.name like '%Yo%'
    AND n1.name not like '%Yu%'
    AND rt.role ='actress'
    AND an1.person_id = n1.id
    AND n1.id = ci.person_id
    AND ci.movie_id = t.id
    AND t.id = mc.movie_id
    AND mc.company_id = cn.id
    AND ci.role_id = rt.id
    AND an1.person_id = ci.person_id
    AND ci.movie_id = mc.movie_id;