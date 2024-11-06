from CapSpot import ImageMatcher
if __name__ == '__main__':
    matcher = ImageMatcher()
    with open('./test_pic/test4_mo.png', 'rb') as template_file, open('./test_pic/test4.png', 'rb') as target_file:
        template_image_data = template_file.read()
        target_image_data = target_file.read()
    matches = matcher.match_images(template_image_data, target_image_data)
    result_stream = matcher.display_results(target_image_data, matches)
    with open('result.png', 'wb') as f:
        f.write(result_stream.read())
    print(matches)