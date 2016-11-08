local classToNumber = function(className) 
    if ( className == 'aeroplane') then
        return 1
    end
    if ( className == 'bicycle') then
        return 2
    end
    if ( className == 'bird' ) then
        return 3
    end
    if ( className == 'boat' ) then
        return 4
    end
    if ( className == 'bottle') then
        return 5
    end
    if ( className == 'bus') then
        return 6
    end
    if ( className == 'car' ) then
        return 7
    end
    if ( className == 'cat' ) then
        return 8
    end
    if ( className == 'chair') then
        return 9
    end
    if ( className == 'cow' ) then
        return 10
    end
    if ( className == 'diningtable' ) then
        return 11
    end
    if ( className == 'dog' ) then
        return 12
    end
    if ( className == 'horse') then
        return 13
    end
    if ( className == 'motorbike') then
        return 14
    end
    if ( className == 'person') then
        return 15
    end
    if ( className == 'pottedplant') then
        return 16
    end
    if ( className == 'sheep') then
        return 17
    end
    if ( className == 'sofa') then
        return 18
    end
    if ( className == 'train') then
        return 19
    end
    if ( className == 'tvmonitor') then
        return 20
    end

    print("   ####### " .. className)

    return 21
end

return classToNumber

